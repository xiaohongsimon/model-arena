import os
import tempfile
import yaml
import pytest
from model_arena.config import load_config, validate_config, ModelConfig


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)


@pytest.fixture
def valid_yaml(tmp_path):
    data = {
        "models": {
            "model-a": {
                "endpoint": "http://127.0.0.1:8002/v1",
                "model_name": "test-model",
                "api_key": "key",
                "timeout_s": 60,
                "max_input_tokens": 32000,
                "max_output_tokens": 4000,
                "status": "active",
            },
            "model-b": {
                "endpoint": "http://127.0.0.1:8002/v1",
                "model_name": "test-model-2",
                "api_key": "key",
                "timeout_s": 90,
                "max_input_tokens": 64000,
                "max_output_tokens": 4000,
                "status": "retired",
            },
        }
    }
    path = tmp_path / "models.yaml"
    _write_yaml(str(path), data)
    return str(path)


def test_load_config_returns_model_configs(valid_yaml):
    models = load_config(valid_yaml)
    assert "model-a" in models
    assert "model-b" in models
    cfg = models["model-a"]
    assert isinstance(cfg, ModelConfig)
    assert cfg.endpoint == "http://127.0.0.1:8002/v1"
    assert cfg.timeout_s == 60
    assert cfg.status == "active"


def test_load_config_filters_active(valid_yaml):
    models = load_config(valid_yaml, active_only=True)
    assert "model-a" in models
    assert "model-b" not in models


def test_load_config_mtime_cache(valid_yaml):
    models1 = load_config(valid_yaml)
    models2 = load_config(valid_yaml)
    assert models1 is models2  # same object, not reloaded


def test_load_config_reloads_on_change(valid_yaml):
    models1 = load_config(valid_yaml)
    # Touch file to change mtime
    os.utime(valid_yaml, (os.path.getmtime(valid_yaml) + 1,) * 2)
    models2 = load_config(valid_yaml)
    assert models1 is not models2


def test_validate_config_missing_field(tmp_path):
    data = {
        "models": {
            "bad-model": {
                "endpoint": "http://127.0.0.1:8002/v1",
                # missing model_name, api_key, etc.
            }
        }
    }
    path = tmp_path / "models.yaml"
    _write_yaml(str(path), data)
    errors = validate_config(str(path))
    assert len(errors) > 0
    assert any("model_name" in e for e in errors)


def test_validate_config_invalid_status(tmp_path):
    data = {
        "models": {
            "bad-model": {
                "endpoint": "http://127.0.0.1:8002/v1",
                "model_name": "test",
                "api_key": "key",
                "timeout_s": 60,
                "max_input_tokens": 32000,
                "max_output_tokens": 4000,
                "status": "broken",
            }
        }
    }
    path = tmp_path / "models.yaml"
    _write_yaml(str(path), data)
    errors = validate_config(str(path))
    assert any("status" in e for e in errors)


def test_validate_config_ok(valid_yaml):
    errors = validate_config(valid_yaml)
    assert errors == []
