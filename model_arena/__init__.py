"""Model Arena: Local multi-model best-of-N competition framework."""
import asyncio
import hashlib
import json
import os
import uuid
from typing import Optional

from model_arena.types import TaskSpec, ArenaResult, ModelOutput, ModelFailure
from model_arena.config import load_config
from model_arena.bandit import pairwise_updates, select_models, compute_stats
from model_arena.storage import ArenaStorage
from model_arena import pool as pool
from model_arena import judge as judge


def _get_config_path() -> str:
    return os.environ.get("ARENA_CONFIG", os.path.join(os.path.dirname(__file__), "..", "models.yaml"))


def _get_db_path() -> str:
    return os.environ.get("ARENA_DB", os.path.join(os.path.dirname(__file__), "..", "arena.db"))


def _get_judge_api_key() -> str:
    return os.environ.get("ARENA_JUDGE_API_KEY", "")


async def compete(
    task_spec: TaskSpec,
    user_prompt: str,
    metadata: Optional[dict] = None,
) -> ArenaResult:
    """Run a model arena competition (async)."""
    run_id = str(uuid.uuid4())[:12]
    metadata = metadata or {}
    prompt_hash = hashlib.sha256(user_prompt.encode()).hexdigest()[:16]

    # 1. Load config
    config_path = _get_config_path()
    all_models = load_config(config_path, active_only=True)

    # 2. Filter eligible models by context window
    prompt_tokens = pool.estimate_tokens(task_spec.system_prompt + user_prompt)
    eligible, skipped = pool.filter_eligible(all_models, prompt_tokens)

    # 3. Get bandit history and select models
    storage = ArenaStorage(_get_db_path())
    await storage.init()

    try:
        records = await storage.get_bandit_history(task_spec.task_type)
        history = compute_stats(records)
        selected_ids = select_models(
            list(eligible.keys()), n=task_spec.default_n, history=history
        )
        selected_configs = [eligible[m] for m in selected_ids]

        # 4. Run parallel inference
        results = await asyncio.gather(*[
            pool.run_inference(cfg, task_spec.system_prompt, user_prompt)
            for cfg in selected_configs
        ])

        # 5. Classify results
        successes = [r for r in results if r["status"] == "success"]
        failures_raw = [r for r in results if r["status"] != "success"]

        # Build failure objects (including skipped)
        failures = []
        for mid in skipped:
            failures.append(ModelFailure(
                model_id=mid, status="skipped_context_limit",
            ))
        for r in failures_raw:
            failures.append(ModelFailure(
                model_id=r["model_id"], status=r["status"],
                error_type=r.get("error_type"), error_message=r.get("error_message"),
                latency_ms=r.get("latency_ms", 0),
            ))

        # Save initial run record
        run_data = {
            "id": run_id,
            "task_type": task_spec.task_type,
            "judge_rubric_version": task_spec.judge_rubric_version,
            "prompt_hash": prompt_hash,
            "metadata_json": json.dumps(metadata),
            "requested_n": task_spec.default_n,
            "eligible_n": len(eligible),
            "selected_n": len(selected_ids),
            "success_count": len(successes),
            "degraded": 0,
            "all_poor": 0,
            "fallback_used": 0,
            "bandit_updated": 0,
            "judge_status": "not_run",
            "status": "running",
        }

        # 6. Handle degraded case: fewer than 2 successes
        if len(successes) < 2:
            run_data["degraded"] = 1
            run_data["degraded_reason"] = f"only {len(successes)} successes"
            run_data["status"] = "completed"

            best_output = None
            winner_model_id = None
            fallback_used = False

            if len(successes) == 1 and task_spec.fallback_policy == "best_single_success":
                best_output = successes[0]["output"]
                winner_model_id = successes[0]["model_id"]
                fallback_used = True
                run_data["fallback_used"] = 1
                run_data["winner_model_id"] = winner_model_id

            run_data["judge_status"] = "skipped"
            await storage.save_run(run_data)

            # Save attempt records
            attempt_records = []
            for r in results:
                attempt_records.append({
                    "run_id": run_id,
                    "model_id": r["model_id"],
                    "status": r["status"],
                    "latency_ms": r.get("latency_ms", 0),
                    "output": r.get("output"),
                    "is_winner": 1 if r["model_id"] == winner_model_id else 0,
                })
            await storage.save_attempts(attempt_records)

            outputs = [
                ModelOutput(
                    model_id=s["model_id"], label="", output=s["output"],
                    latency_ms=s.get("latency_ms", 0),
                ) for s in successes
            ]

            return ArenaResult(
                run_id=run_id,
                best_output=best_output,
                winner_model_id=winner_model_id,
                winner_score=None,
                judge_comment="",
                outputs=outputs,
                failures=failures,
                degraded=True,
                degraded_reason=run_data["degraded_reason"],
                fallback_used=fallback_used,
                bandit_updated=False,
            )

        # 7. Judge the outputs
        success_ids = [r["model_id"] for r in successes]
        id_to_label, label_to_id = judge.randomize_labels(success_ids)

        label_outputs = {}
        for r in successes:
            label_outputs[id_to_label[r["model_id"]]] = r["output"]

        judge_result = await judge.judge_outputs(
            task_spec, user_prompt, label_outputs, _get_judge_api_key(),
        )

        run_data["label_mapping_json"] = json.dumps(label_to_id)
        run_data["judge_raw_response"] = judge_result.get("raw_response", "")
        run_data["judge_status"] = judge_result["status"]

        if judge_result["status"] != "completed" or judge_result["parsed"] is None:
            # Judge failed
            run_data["degraded"] = 1
            run_data["degraded_reason"] = "judge_error"
            run_data["status"] = "completed"
            await storage.save_run(run_data)

            # Fallback: pick first success
            best = successes[0]
            outputs = [
                ModelOutput(
                    model_id=s["model_id"], label=id_to_label[s["model_id"]],
                    output=s["output"], latency_ms=s.get("latency_ms", 0),
                ) for s in successes
            ]
            attempt_records = []
            for r in results:
                attempt_records.append({
                    "run_id": run_id,
                    "model_id": r["model_id"],
                    "label": id_to_label.get(r["model_id"]),
                    "status": r["status"],
                    "latency_ms": r.get("latency_ms", 0),
                    "output": r.get("output"),
                    "is_winner": 0,
                })
            await storage.save_attempts(attempt_records)

            return ArenaResult(
                run_id=run_id,
                best_output=best["output"],
                winner_model_id=best["model_id"],
                winner_score=None,
                judge_comment="judge_error",
                outputs=outputs,
                failures=failures,
                degraded=True,
                degraded_reason="judge_error",
                fallback_used=True,
                bandit_updated=False,
            )

        # 8. Process judge results
        parsed = judge_result["parsed"]
        scores = parsed["scores"]
        all_poor = parsed.get("all_poor", False)
        winner_label = parsed.get("winner")
        comment = parsed.get("comment", "")

        run_data["judge_comment"] = comment
        run_data["all_poor"] = 1 if all_poor else 0

        # Pairwise updates (only if not all_poor)
        bandit_updated = False
        wins_map = {}
        losses_map = {}
        if not all_poor:
            wins_map, losses_map = pairwise_updates(scores, task_spec.tie_epsilon)
            bandit_updated = True

        run_data["bandit_updated"] = 1 if bandit_updated else 0

        # Determine winner
        winner_model_id = None
        winner_score = None
        best_output = None
        if winner_label and winner_label in label_to_id:
            winner_model_id = label_to_id[winner_label]
            winner_score = scores.get(winner_label)
            for s in successes:
                if s["model_id"] == winner_model_id:
                    best_output = s["output"]
                    break

        run_data["winner_label"] = winner_label
        run_data["winner_model_id"] = winner_model_id
        run_data["winner_score"] = winner_score
        run_data["status"] = "completed"
        await storage.save_run(run_data)

        # Save attempts with scores and pairwise data
        attempt_records = []
        outputs = []
        for r in results:
            mid = r["model_id"]
            label = id_to_label.get(mid)
            att = {
                "run_id": run_id,
                "model_id": mid,
                "label": label,
                "status": r["status"],
                "latency_ms": r.get("latency_ms", 0),
                "output": r.get("output"),
                "is_winner": 1 if mid == winner_model_id else 0,
            }
            if r["status"] == "success" and label in scores:
                att["score"] = scores[label]
                att["pairwise_wins"] = wins_map.get(label, 0.0)
                att["pairwise_losses"] = losses_map.get(label, 0.0)
                outputs.append(ModelOutput(
                    model_id=mid, label=label, output=r["output"],
                    score=scores[label],
                    pairwise_wins=wins_map.get(label, 0.0),
                    pairwise_losses=losses_map.get(label, 0.0),
                    latency_ms=r.get("latency_ms", 0),
                ))
            attempt_records.append(att)
        await storage.save_attempts(attempt_records)

        return ArenaResult(
            run_id=run_id,
            best_output=best_output,
            winner_model_id=winner_model_id,
            winner_score=winner_score,
            judge_comment=comment,
            outputs=outputs,
            failures=failures,
            degraded=False,
            degraded_reason=None,
            fallback_used=False,
            bandit_updated=bandit_updated,
        )

    finally:
        await storage.close()


def compete_sync(
    task_spec: TaskSpec,
    user_prompt: str,
    metadata: Optional[dict] = None,
) -> ArenaResult:
    """Sync wrapper for compete()."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g., Jupyter)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                compete(task_spec, user_prompt, metadata),
            )
            return future.result()
    else:
        return asyncio.run(compete(task_spec, user_prompt, metadata))
