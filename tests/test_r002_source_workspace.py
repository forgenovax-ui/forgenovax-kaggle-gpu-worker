import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_source_drift_selects_fresh_workspace_without_deleting_original(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / "run_r002_experiment.py"
    runner.write_text("# frozen source\n", encoding="utf-8")
    shell = ROOT / "scripts" / "run_r002_frozen.sh"
    command = f"""
source {shell!s}
source_tree_sha256="$(source_tree_digest {source!s})"
same="$(select_source_dir {source!s})"
test "$same" = {source!s}
printf '# runtime drift\n' >> {runner!s}
fresh="$(select_source_dir {source!s})"
test "$fresh" != {source!s}
test -d "$fresh"
test -f {runner!s}
"""
    environment = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
        "FORGENOVAX_WORK_DIR": str(tmp_path),
        "FNX_R002_SOURCE_SHA256": "archive",
        "FNX_R002_SOURCE_TREE_SHA256": "initial",
        "FNX_R002_SOURCE_GIT_SHA": "git",
    }
    subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
