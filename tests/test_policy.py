import json

from transcript_lok.policy import load_policy


def test_policy_loads_feature_flags_and_production_defaults(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "features": {"diarization": False, "cloud_providers": False},
                "runtime": {"production_cpu": True},
            }
        ),
        encoding="utf-8",
    )

    policy = load_policy(path)

    assert not policy.feature_enabled("diarization")
    assert not policy.feature_enabled("cloud_providers")
    assert policy.runtime.default_model == "base"
    assert policy.runtime.default_device == "cpu"
    assert policy.runtime.default_compute_type == "int8"
    assert policy.runtime.cpu_threads == 2
    assert policy.runtime.beam_size == 1
    assert policy.runtime.chunk_long_files is True


def test_policy_env_can_enable_production_cpu(monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_LOK_PRODUCTION_CPU", "1")

    policy = load_policy()

    assert policy.runtime.production_cpu is True
    assert policy.runtime.default_model == "base"
    assert policy.runtime.queue_max_threads == 2
