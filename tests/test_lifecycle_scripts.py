from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_start_ollama_is_idempotent_when_loopback_api_is_healthy(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(
        bin_dir / "nvidia-smi",
        "#!/bin/sh\nprintf '0\\n1\\n'\n",
    )
    executable(bin_dir / "curl", "#!/bin/sh\nexit 0\n")
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "FORGENOVAX_WORK_DIR": str(tmp_path),
            "FORGENOVAX_REQUIRE_GPU": "1",
            "OLLAMA_HOST": "127.0.0.1:11434",
        }
    )

    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "start_ollama.sh")],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "already ready" in result.stdout
    assert not (tmp_path / "ollama.pid").exists()


def test_shutdown_refuses_to_kill_reused_pid_and_removes_stale_file(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "cloudflared.pid"
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PATH": "/usr/bin:/bin",
            "FORGENOVAX_WORK_DIR": str(tmp_path),
        }
    )

    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "shutdown.sh")],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "refusing to kill" in result.stdout
    assert not pid_file.exists()
