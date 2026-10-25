from pathlib import Path
from hashlib import sha256
import json
import shutil

import pytest

from src.model_cache import export_cache, restore_cache, verify_cache


def sample_store(tmp_path: Path) -> Path:
    store = tmp_path / "original"
    (store / "blobs").mkdir(parents=True)
    (store / "manifests" / "registry.ollama.ai" / "library" / "tiny").mkdir(parents=True)
    (store / "blobs" / "sha256-synthetic").write_bytes(b"fake model weights only")
    (store / "manifests" / "registry.ollama.ai" / "library" / "tiny" / "latest").write_text('{"schemaVersion":2}')
    return store


def test_export_verify_restore_roundtrip(tmp_path: Path) -> None:
    source = sample_store(tmp_path)
    snapshot = tmp_path / "export"
    dest = tmp_path / "restored"
    manifest = export_cache(source, snapshot)
    assert manifest["format"] == "forgenovax-ollama-cache-v1"
    assert len(manifest["files"]) == 2
    assert verify_cache(snapshot) == manifest
    assert restore_cache(snapshot, dest) == manifest
    assert (dest / "blobs" / "sha256-synthetic").read_bytes() == b"fake model weights only"


def test_corrupt_blob_is_rejected_without_writing_target(tmp_path: Path) -> None:
    snapshot = tmp_path / "export"
    export_cache(sample_store(tmp_path), snapshot)
    (snapshot / "ollama-models" / "blobs" / "sha256-synthetic").write_bytes(b"tampered")
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="mismatch"):
        restore_cache(snapshot, target)
    assert not target.exists()


def test_extra_file_is_rejected(tmp_path: Path) -> None:
    snapshot = tmp_path / "export"
    export_cache(sample_store(tmp_path), snapshot)
    (snapshot / "ollama-models" / "unexpected").write_text("new")
    with pytest.raises(ValueError, match="mismatch"):
        verify_cache(snapshot)


def test_existing_target_is_never_overwritten(tmp_path: Path) -> None:
    snapshot = tmp_path / "export"
    export_cache(sample_store(tmp_path), snapshot)
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep-me").write_text("important")
    with pytest.raises(FileExistsError, match="Refusing"):
        restore_cache(snapshot, target)
    assert (target / "keep-me").read_text() == "important"


def test_export_rejects_nested_destination(tmp_path: Path) -> None:
    store = sample_store(tmp_path)
    with pytest.raises(ValueError, match="must not be inside"):
        export_cache(store, store / "nested")


def test_unexpected_symlinks_not_exported(tmp_path: Path) -> None:
    store = sample_store(tmp_path)
    (store / "blobs" / "other").symlink_to(store / "blobs" / "sha256-synthetic")
    with pytest.raises(ValueError, match="Unsafe"):
        export_cache(store, tmp_path / "export")


def test_current_ollama_manifest_symlink_roundtrip(tmp_path: Path) -> None:
    store = sample_store(tmp_path)
    manifest_blob = store / "blobs" / ("sha256-" + "a" * 64)
    manifest_blob.write_text('{"schemaVersion":2}')
    v2 = store / "manifests-v2" / "ollama.com" / "library" / "tiny"
    v2.mkdir(parents=True)
    (v2 / "latest").symlink_to(Path("../../../../blobs") / manifest_blob.name)

    snapshot = tmp_path / "export"
    restored = tmp_path / "restored"
    manifest = export_cache(store, snapshot)
    assert manifest["links"] == [{
        "path": "manifests-v2/ollama.com/library/tiny/latest",
        "target": f"../../../../blobs/{manifest_blob.name}",
    }]
    restore_cache(snapshot, restored)
    restored_link = restored / "manifests-v2" / "ollama.com" / "library" / "tiny" / "latest"
    assert restored_link.is_symlink()
    assert restored_link.resolve().read_text() == '{"schemaVersion":2}'


def test_manifest_symlink_cannot_escape_store(tmp_path: Path) -> None:
    store = sample_store(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("not model data")
    v2 = store / "manifests-v2"
    v2.mkdir()
    (v2 / "unsafe").symlink_to(outside)
    with pytest.raises(ValueError, match="Unsafe"):
        export_cache(store, tmp_path / "export")


def test_no_snapshot_overwrite(tmp_path: Path) -> None:
    source = sample_store(tmp_path)
    snapshot = tmp_path / "export"
    export_cache(source, snapshot)
    with pytest.raises(FileExistsError, match="overwrite"):
        export_cache(source, snapshot)


def test_manifest_with_missing_referenced_blob_is_rejected(tmp_path: Path) -> None:
    store = sample_store(tmp_path)
    digest = sha256(b"missing model layer").hexdigest()
    manifest = store / "manifests" / "registry.ollama.ai" / "library" / "tiny" / "latest"
    manifest.write_text(json.dumps({
        "schemaVersion": 2,
        "layers": [{"digest": f"sha256:{digest}"}],
    }))
    with pytest.raises(ValueError, match="missing or unsafe blob"):
        export_cache(store, tmp_path / "export")


def test_interrupted_restore_cleans_staging_without_target(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = tmp_path / "export"
    export_cache(sample_store(tmp_path), snapshot)
    target = tmp_path / "target"

    def interrupted_copytree(source: Path, destination: Path, **_: object) -> None:
        destination.mkdir(parents=True)
        (destination / "partial").write_text("incomplete")
        raise OSError("simulated interruption")

    monkeypatch.setattr(shutil, "copytree", interrupted_copytree)
    with pytest.raises(OSError, match="interruption"):
        restore_cache(snapshot, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".target.restore-*"))
