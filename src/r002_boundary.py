"""Fail-closed authorization boundary for the FNX-R002 held-out evaluation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


REQUIRED_TRUE = (
    "public_stream_live",
    "youtube_ingest_healthy",
    "obs_recording_active",
    "public_dashboard_healthy",
    "sanitizer_pass",
    "experiment_manifest_frozen",
)


def _timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a UTC offset")
    return parsed


def authorize(
    payload: dict[str, Any],
    *,
    not_before: datetime,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate every public-live gate and record the held-out transition."""
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("now must include a UTC offset")
    if moment < not_before:
        raise ValueError("scheduled public start has not been reached")
    if payload.get("experiment_id") != "FNX-R002":
        raise ValueError("experiment_id must be FNX-R002")
    failed = [field for field in REQUIRED_TRUE if payload.get(field) is not True]
    if failed:
        raise ValueError("live boundary gates failed: " + ", ".join(failed))
    if payload.get("held_out_accessed") is not False:
        raise ValueError("held_out_accessed must be false before authorization")
    confirmed_at = _timestamp(str(payload.get("stream_health_confirmed_at", "")), field="stream_health_confirmed_at")
    if confirmed_at < not_before:
        raise ValueError("stream health was confirmed before the scheduled public start")
    if confirmed_at > moment:
        raise ValueError("stream health confirmation cannot be in the future")
    result = dict(payload)
    result["held_out_accessed"] = True
    result["held_out_accessed_at"] = moment.astimezone(UTC).isoformat()
    result["authorization_status"] = "AUTHORIZED"
    result["production_status"] = "NOT_PRODUCTION_CERTIFIED"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--not-before", required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("boundary manifest must be a JSON object")
        not_before = _timestamp(args.not_before, field="not_before")
        authorized = authorize(payload, not_before=not_before)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FNX-R002 HELD-OUT BOUNDARY: BLOCKED ({exc})")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(authorized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print("FNX-R002 HELD-OUT BOUNDARY: AUTHORIZED")
    print(f"HELD_OUT_ACCESSED_AT={authorized['held_out_accessed_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
