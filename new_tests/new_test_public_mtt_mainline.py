import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_mainline_config():
    config = json.loads(
        (
            ROOT
            / "new_configs/new_public_mtt_mainline/new_mainline_config.json"
        ).read_text(encoding="utf-8")
    )
    assert config["stage_a_epsilon"] == "N/A"
    assert config["stage_a_dp_protected"] is False
    assert config["dp_mtt_status"] == "EXCLUDED_BY_USER"
    assert config["feature_dim"] == 2048
    assert config["cpu_thread_policy"] == "benchmark_max_throughput"
    assert config["num_workers_policy"] == "benchmark_max_throughput"
    assert config["test_tuning_forbidden"] is True


def test_tuning_cache_path_is_separate():
    source = (ROOT / "new_public_mtt_mainline_common.py").read_text(
        encoding="utf-8"
    )
    assert "new_test_free_features" in source
    assert "test-free tuning cache contains test data" in source
