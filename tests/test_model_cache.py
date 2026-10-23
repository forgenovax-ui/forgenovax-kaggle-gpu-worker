from pathlib import Path

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


def test_symlinks_not_exported(tmp_path: Path) -> None:
    store = sample_store(tmp_path)
    (store / "blobs" / "other").symlink_to(store / "blobs" / "sha256-synthetic")
    with pytest.raises(ValueError, match="Symlinks"):
        export_cache(store, tmp_path / "export")


def test_no_snapshot_overwrite(tmp_path: Path) -> None:
    source = sample_store(tmp_path)
    snapshot = tmp_path / "export"
    export_cache(source, snapshot)
    with pytest.raises(FileExistsError, match="overwrite"):
        export_cache(source, snapshot)
