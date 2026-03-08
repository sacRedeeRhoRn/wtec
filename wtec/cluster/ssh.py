"""SSH/SFTP connection management using paramiko."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wtec.config.cluster import ClusterConfig


class SSHClient:
    """Thin wrapper around paramiko SSHClient with SFTP support."""

    def __init__(self, client, sftp=None) -> None:
        self._client = client
        self._sftp = sftp

    def run(self, command: str, *, check: bool = True) -> tuple[int, str, str]:
        """Execute command on remote. Returns (returncode, stdout, stderr)."""
        stdin, stdout, stderr = self._client.exec_command(command)
        rc = stdout.channel.recv_exit_status()
        out = stdout.read().decode()
        err = stderr.read().decode()
        if check and rc != 0:
            raise RuntimeError(
                f"Remote command failed (rc={rc}):\n  $ {command}\n  stderr: {err.strip()}"
            )
        return rc, out, err

    def put(self, local: str | Path, remote: str) -> None:
        """Upload a file."""
        self._sftp.put(str(local), remote)

    def get(self, remote: str, local: str | Path) -> None:
        """Download a file."""
        self._sftp.get(remote, str(local))

    def get_many(self, remote_dir: str, local_dir: str | Path, patterns: list[str]) -> None:
        """Download files matching glob patterns from remote_dir."""
        import fnmatch
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        items = self._sftp.listdir(remote_dir)
        for item in items:
            for pat in patterns:
                if fnmatch.fnmatch(item, pat):
                    self.get(
                        f"{remote_dir}/{item}",
                        local_dir / item,
                    )
                    break

    def mkdir_p(self, remote_path: str) -> None:
        """Create remote directory (including parents) if not exists."""
        parts = remote_path.rstrip("/").split("/")
        path = ""
        for part in parts:
            if not part:
                path = "/"
                continue
            path = path.rstrip("/") + "/" + part
            try:
                self._sftp.mkdir(path)
            except OSError:
                pass  # already exists

    def close(self) -> None:
        if self._sftp:
            self._sftp.close()
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@contextlib.contextmanager
def open_ssh(cfg: "ClusterConfig"):
    """Context manager that opens an SSH+SFTP connection.

    Usage::

        with open_ssh(ClusterConfig.from_env()) as ssh:
            ssh.run("qstat")

    Credentials are taken from ClusterConfig, never hardcoded.
    """
    try:
        import paramiko
    except ImportError:
        raise ImportError("paramiko is required: pip install paramiko")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs: dict = dict(
        hostname=cfg.host,
        port=cfg.port,
        username=cfg.user,
        timeout=30,
    )
    key_file = Path(cfg.key_path).expanduser() if cfg.key_path else None
    if key_file and key_file.exists():
        connect_kwargs["key_filename"] = str(key_file)
    elif cfg.password:
        connect_kwargs["password"] = cfg.password
    elif key_file and not key_file.exists():
        raise ValueError(
            "TOPOSLAB_CLUSTER_KEY is set but file does not exist: "
            f"{key_file}. Set a valid key path or TOPOSLAB_CLUSTER_PASS."
        )
    else:
        raise ValueError(
            "No authentication method configured. "
            "Set TOPOSLAB_CLUSTER_PASS or TOPOSLAB_CLUSTER_KEY."
        )

    client.connect(**connect_kwargs)
    sftp = client.open_sftp()
    wrapped = SSHClient(client, sftp)
    try:
        yield wrapped
    finally:
        wrapped.close()
