"""Portable, integrity-checked Ollama cache snapshots for notebook reuse.

Export produces an ordinary directory that can be saved as Kaggle notebook
output or as a private Dataset. Restore requires an explicitly mounted input.
No API token or Kaggle credentials are needed.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

FORMAT = "forgenovax-ollama-cache-v1"
MANIFEST = "model-cache.json"
CONTENT_DIR = "ollama-models"


def _files(directory: Path) -> list[dict[str, object]]:
    if not directory.is_dir():
        raise ValueError(f"Model directory does not exist: {directory}")
    result: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not permitted in model caches: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        digest = sha256()
        size = 0
        with path.open("rb") as reader:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        result.append({"path": relative, "bytes": size, "sha256": digest.hexdigest()})
    return result


def _validate_layout(directory: Path) -> None:
    if not (directory / "blobs").is_dir() or not (directory / "manifests").is_dir():
        raise ValueError("Expected an Ollama model store with blobs/ and manifests/")
    if not _files(directory):
        raise ValueError("Model store is empty")


def verify_cache(cache: Path) -> dict[str, object]:
    manifest_path = cache / MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cache manifest is absent or malformed") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError("Unsupported cache manifest format")
    directory = cache / CONTENT_DIR
    _validate_layout(directory)
    actual = _files(directory)
    if actual != manifest.get("files"):
        raise ValueError("Cache checksum, file count or file size mismatch")
    return manifest


def export_cache(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    output = output.expanduser().resolve()
    _validate_layout(source)
    if output == source or source in output.parents:
        raise ValueError("Output directory must not be inside the model store")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(source, output / CONTENT_DIR)
        files = _files(output / CONTENT_DIR)
        manifest: dict[str, object] = {
            "format": FORMAT,
            "files": files,
            "total_bytes": sum(int(f["bytes"]) for f in files),
        }
        (output / MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_cache(output)
        return manifest
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def restore_cache(cache: Path, target: Path) -> dict[str, object]:
    cache = cache.resolve()
    target = target.expanduser().resolve()
    manifest = verify_cache(cache)
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing model store: {target}. "
            "Choose a clean target or leave the current cache in place."
        )
    if target == cache or cache in target.parents:
        raise ValueError("Model store target must not be inside the input cache")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.restore-{uuid4().hex}"
    try:
        shutil.copytree(cache / CONTENT_DIR, staging)
        if _files(staging) != manifest["files"]:
            raise ValueError("Restored files failed checksum verification")
        staging.rename(target)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Create a portable model-cache snapshot")
    export.add_argument("--source", default=os.getenv("OLLAMA_MODELS", "/kaggle/working/ollama-models"))
    export.add_argument("--output", default="/kaggle/working/fnx-cache-export")
    restore = sub.add_parser("restore", help="Restore a mounted, verified model-cache snapshot")
    restore.add_argument("--input", required=True)
    restore.add_argument("--target", default=os.getenv("OLLAMA_MODELS", "/kaggle/working/ollama-models"))
    verify = sub.add_parser("verify", help="Check the checksum manifest of an input cache")
    verify.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            result = export_cache(Path(args.source), Path(args.output))
            print(f"Cache export: {args.output}")
        elif args.command == "restore":
            result = restore_cache(Path(args.input), Path(args.target))
            print(f"Cache restored: {args.target}")
        else:
            result = verify_cache(Path(args.input))
            print(f"Cache verified: {args.input}")
        print(f"Files: {len(result['files'])}; bytes: {result['total_bytes']}")
        return 0
    except (ValueError, FileExistsError, OSError) as exc:
        print(f"MODEL_CACHE_ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
