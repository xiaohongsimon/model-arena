import aiosqlite
from typing import Optional

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS arena_runs (
    id TEXT PRIMARY KEY,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    task_type TEXT NOT NULL,
    judge_rubric_version TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    requested_n INTEGER NOT NULL,
    eligible_n INTEGER NOT NULL,
    selected_n INTEGER NOT NULL,
    success_count INTEGER NOT NULL DEFAULT 0,
    label_mapping_json TEXT,
    judge_model TEXT,
    judge_raw_response TEXT,
    judge_status TEXT NOT NULL DEFAULT 'not_run',
    winner_label TEXT,
    winner_model_id TEXT,
    winner_score REAL,
    judge_comment TEXT,
    all_poor INTEGER NOT NULL DEFAULT 0,
    degraded INTEGER NOT NULL DEFAULT 0,
    degraded_reason TEXT,
    fallback_used INTEGER NOT NULL DEFAULT 0,
    bandit_updated INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS arena_attempts (
    run_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    label TEXT,
    status TEXT NOT NULL,
    latency_ms INTEGER,
    output TEXT,
    score REAL,
    pairwise_wins REAL,
    pairwise_losses REAL,
    is_winner INTEGER NOT NULL DEFAULT 0,
    error_type TEXT,
    error_message TEXT,
    PRIMARY KEY (run_id, model_id),
    FOREIGN KEY (run_id) REFERENCES arena_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_runs_task_ts ON arena_runs(task_type, ts DESC);
CREATE INDEX IF NOT EXISTS idx_attempts_model_status ON arena_attempts(model_id, status);
"""


class ArenaStorage:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def init(self):
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)

    async def close(self):
        if self._conn:
            await self._conn.close()

    async def fetch_all(self, sql: str, params=()) -> list:
        cursor = await self._conn.execute(sql, params)
        return await cursor.fetchall()

    async def save_run(self, data: dict):
        cols = list(data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        await self._conn.execute(
            f"INSERT INTO arena_runs ({col_str}) VALUES ({placeholders})",
            [data[c] for c in cols],
        )
        await self._conn.commit()

    async def update_run(self, run_id: str, data: dict):
        sets = ", ".join(f"{k} = ?" for k in data)
        vals = list(data.values()) + [run_id]
        await self._conn.execute(f"UPDATE arena_runs SET {sets} WHERE id = ?", vals)
        await self._conn.commit()

    async def get_run(self, run_id: str) -> Optional[dict]:
        cursor = await self._conn.execute("SELECT * FROM arena_runs WHERE id = ?", (run_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def save_attempts(self, attempts: list[dict]):
        for att in attempts:
            cols = list(att.keys())
            placeholders = ", ".join(["?"] * len(cols))
            col_str = ", ".join(cols)
            await self._conn.execute(
                f"INSERT INTO arena_attempts ({col_str}) VALUES ({placeholders})",
                [att[c] for c in cols],
            )
        await self._conn.commit()

    async def get_attempts(self, run_id: str) -> list[dict]:
        cursor = await self._conn.execute(
            "SELECT * FROM arena_attempts WHERE run_id = ?", (run_id,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_bandit_history(self, task_type: str) -> list[tuple[str, float, float]]:
        sql = """
            SELECT a.model_id, a.pairwise_wins, a.pairwise_losses
            FROM arena_attempts a
            JOIN arena_runs r ON a.run_id = r.id
            WHERE r.task_type = ?
              AND r.bandit_updated = 1
              AND a.status = 'success'
              AND a.pairwise_wins IS NOT NULL
        """
        cursor = await self._conn.execute(sql, (task_type,))
        rows = await cursor.fetchall()
        return [(r[0], r[1], r[2]) for r in rows]
