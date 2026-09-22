from datetime import datetime

import pytest

from src.r002_boundary import authorize


NOT_BEFORE = datetime.fromisoformat("2026-09-22T09:00:00-04:00")
NOW = datetime.fromisoformat("2026-09-22T09:10:00-04:00")


def ready_payload() -> dict[str, object]:
    return {
        "experiment_id": "FNX-R002",
        "public_stream_live": True,
        "youtube_ingest_healthy": True,
        "obs_recording_active": True,
        "public_dashboard_healthy": True,
        "sanitizer_pass": True,
        "experiment_manifest_frozen": True,
        "stream_health_confirmed_at": "2026-09-22T09:08:00-04:00",
        "held_out_accessed": False,
    }


def test_authorizes_only_after_all_live_gates() -> None:
    result = authorize(ready_payload(), not_before=NOT_BEFORE, now=NOW)

    assert result["authorization_status"] == "AUTHORIZED"
    assert result["held_out_accessed"] is True
    assert result["held_out_accessed_at"] == "2026-09-22T13:10:00+00:00"


@pytest.mark.parametrize("field", [
    "public_stream_live",
    "youtube_ingest_healthy",
    "obs_recording_active",
    "public_dashboard_healthy",
    "sanitizer_pass",
    "experiment_manifest_frozen",
])
def test_blocks_each_missing_live_gate(field: str) -> None:
    payload = ready_payload()
    payload[field] = False

    with pytest.raises(ValueError, match=field):
        authorize(payload, not_before=NOT_BEFORE, now=NOW)


def test_blocks_before_scheduled_public_start() -> None:
    with pytest.raises(ValueError, match="scheduled public start"):
        authorize(
            ready_payload(),
            not_before=NOT_BEFORE,
            now=datetime.fromisoformat("2026-09-22T08:59:59-04:00"),
        )


def test_blocks_prior_held_out_access() -> None:
    payload = ready_payload()
    payload["held_out_accessed"] = True

    with pytest.raises(ValueError, match="must be false"):
        authorize(payload, not_before=NOT_BEFORE, now=NOW)


def test_blocks_health_confirmation_before_public_start() -> None:
    payload = ready_payload()
    payload["stream_health_confirmed_at"] = "2026-09-22T08:59:59-04:00"

    with pytest.raises(ValueError, match="confirmed before"):
        authorize(payload, not_before=NOT_BEFORE, now=NOW)
