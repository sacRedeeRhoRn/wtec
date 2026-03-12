"""PBS job submission, monitoring, and file retrieval."""

from __future__ import annotations

import os
import posixpath
import re
import shlex
import time
from pathlib import Path
from typing import Literal


JobStatus = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "UNKNOWN"]


class JobManager:
    """Submit and track PBS jobs via SSH."""

    DEFAULT_QUEUE_PRIORITY = ["g4", "g3", "g2", "g1"]

    def __init__(self, ssh) -> None:
        """
        Parameters
        ----------
        ssh : wtec.cluster.ssh.SSHClient
        """
        self._ssh = ssh

    def _parse_qsub_job_id(self, output: str) -> str:
        m = re.search(r"(?P<job>\d+)(?:\.[A-Za-z0-9_.-]+)?", output)
        if not m:
            raise RuntimeError(f"Could not parse qsub job ID from output: {output.strip()!r}")
        return m.group("job")

    def _available_queues(self) -> dict[str, dict[str, bool]]:
        """Return queue availability metadata keyed by queue name."""
        rc, stdout, _ = self._ssh.run("qstat -Q -f 2>/dev/null", check=False)
        if rc != 0 or not stdout.strip():
            return {}

        queue_data: dict[str, dict[str, bool]] = {}
        current: str | None = None
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if line.startswith("Queue:"):
                current = line.split(":", 1)[1].strip()
                queue_data[current] = {"enabled": False, "started": False}
                continue
            if current is None:
                continue
            if line.startswith("enabled ="):
                queue_data[current]["enabled"] = line.split("=", 1)[1].strip().lower() == "yes"
            elif line.startswith("started ="):
                queue_data[current]["started"] = line.split("=", 1)[1].strip().lower() == "yes"
        return queue_data

    def resolve_queue(
        self,
        preferred: str | None = None,
        *,
        fallback_order: list[str] | None = None,
    ) -> str:
        """Resolve PBS queue. Explicit queue overrides fallback selection."""
        queue_map = self._available_queues()
        order = fallback_order or self.DEFAULT_QUEUE_PRIORITY
        preferred = (preferred or "").strip() or None

        if preferred:
            if not queue_map:
                return preferred
            info = queue_map.get(preferred)
            if info is None:
                available = ", ".join(sorted(queue_map))
                raise RuntimeError(
                    f"Requested PBS queue {preferred!r} not found. Available queues: {available}"
                )
            if not (info.get("enabled") and info.get("started")):
                raise RuntimeError(
                    f"Requested PBS queue {preferred!r} is not enabled/started."
                )
            return preferred

        if not queue_map:
            raise RuntimeError(
                "Could not query PBS queues via `qstat -Q -f`, and no queue was provided."
            )

        for q in order:
            info = queue_map.get(q)
            if info and info.get("enabled") and info.get("started"):
                return q

        for q in sorted(queue_map):
            info = queue_map[q]
            if info.get("enabled") and info.get("started"):
                return q

        raise RuntimeError("No enabled/started PBS queue is available.")

    def stage_files(
        self,
        local_files: list[str | Path],
        remote_dir: str,
    ) -> list[str]:
        """Upload local files into remote_dir after basic integrity checks."""
        self._ssh.mkdir_p(remote_dir)
        staged: list[str] = []
        for local in local_files:
            p = Path(local)
            if not p.exists():
                raise FileNotFoundError(f"Required local file is missing: {p}")
            if p.stat().st_size == 0:
                raise ValueError(f"Required local file is empty: {p}")
            remote_path = f"{remote_dir.rstrip('/')}/{p.name}"
            self._ssh.put(p, remote_path)
            staged.append(remote_path)
        return staged

    def ensure_remote_files(self, remote_dir: str, filenames: list[str]) -> None:
        """Fail if any required remote files are missing or empty."""
        missing: list[str] = []
        for name in filenames:
            target = f"{remote_dir.rstrip('/')}/{name}"
            rc, _, _ = self._ssh.run(f"test -s {shlex.quote(target)}", check=False)
            if rc != 0:
                missing.append(target)
        if missing:
            raise RuntimeError(
                "Missing required remote file(s):\n  " + "\n  ".join(missing)
            )

    def ensure_remote_commands(
        self,
        commands: list[str],
        *,
        modules: list[str] | None = None,
        bin_dirs: list[str] | None = None,
    ) -> None:
        """Fail if required commands are not in remote PATH."""
        missing: list[str] = []
        resolved_bin_dirs = list(bin_dirs or [])
        if not resolved_bin_dirs:
            raw = os.environ.get("TOPOSLAB_CLUSTER_BIN_DIRS", "").strip()
            if raw:
                resolved_bin_dirs = [p.strip() for p in raw.split(",") if p.strip()]
        module_prefix = (
            " && ".join(
                f"module load {shlex.quote(m)} >/dev/null 2>&1" for m in modules
            )
            if modules
            else ""
        )
        path_prefix = (
            "export PATH="
            + ":".join(shlex.quote(p) for p in resolved_bin_dirs)
            + ":$PATH"
            if resolved_bin_dirs
            else ""
        )
        for cmd in commands:
            command_check = f"command -v {shlex.quote(cmd)}"
            if module_prefix or path_prefix:
                parts = [p for p in [module_prefix, path_prefix, command_check] if p]
                wrapped = " && ".join(parts)
                run_cmd = f"bash -lc {shlex.quote(wrapped)}"
            else:
                run_cmd = command_check
            rc, _, _ = self._ssh.run(run_cmd, check=False)
            if rc != 0:
                missing.append(cmd)
        if missing:
            raise RuntimeError(
                "Missing required executable(s) on cluster PATH: "
                + ", ".join(missing)
            )

    def ensure_remote_mpi_binaries(
        self,
        commands: list[str],
        *,
        modules: list[str] | None = None,
        bin_dirs: list[str] | None = None,
    ) -> None:
        """Fail if executable appears to be non-MPI linked.

        This guards against launching serial-only binaries under mpirun, which
        can spawn many independent serial instances and hang on shared stdin.
        """
        import os

        # Allow explicit opt-out for unusual static-linked MPI binaries.
        if os.environ.get("TOPOSLAB_SKIP_MPI_BINARY_CHECK", "").strip().lower() in {"1", "true", "yes", "on"}:
            return

        resolved_bin_dirs = list(bin_dirs or [])
        if not resolved_bin_dirs:
            raw = os.environ.get("TOPOSLAB_CLUSTER_BIN_DIRS", "").strip()
            if raw:
                resolved_bin_dirs = [p.strip() for p in raw.split(",") if p.strip()]
        module_prefix = (
            " && ".join(
                f"module load {shlex.quote(m)} >/dev/null 2>&1" for m in modules
            )
            if modules
            else ""
        )
        path_prefix = (
            "export PATH="
            + ":".join(shlex.quote(p) for p in resolved_bin_dirs)
            + ":$PATH"
            if resolved_bin_dirs
            else ""
        )

        serial_like: list[str] = []
        for cmd in commands:
            check = (
                "resolved=$(command -v "
                + shlex.quote(cmd)
                + ") && "
                "[ -n \"$resolved\" ] && [ -x \"$resolved\" ] && "
                "ldd \"$resolved\" 2>/dev/null | "
                "grep -Eiq '(libmpi|openmpi|mpich|open-rte)'"
            )
            parts = [p for p in [module_prefix, path_prefix, check] if p]
            wrapped = " && ".join(parts)
            run_cmd = f"bash -lc {shlex.quote(wrapped)}"
            rc, _, _ = self._ssh.run(run_cmd, check=False)
            if rc != 0:
                serial_like.append(cmd)
        if serial_like:
            raise RuntimeError(
                "Executable(s) appear to be non-MPI linked: "
                + ", ".join(serial_like)
                + ". Running these under mpirun can hang or duplicate serial runs. "
                "Install/build MPI-enabled binaries or set TOPOSLAB_SKIP_MPI_BINARY_CHECK=1 to override."
            )

    def ensure_remote_python_imports(
        self,
        python_executable: str,
        modules_to_import: list[str],
        *,
        modules: list[str] | None = None,
        bin_dirs: list[str] | None = None,
    ) -> None:
        """Fail if the remote Python cannot import the requested modules."""
        resolved_bin_dirs = list(bin_dirs or [])
        if not resolved_bin_dirs:
            raw = os.environ.get("TOPOSLAB_CLUSTER_BIN_DIRS", "").strip()
            if raw:
                resolved_bin_dirs = [p.strip() for p in raw.split(",") if p.strip()]
        module_prefix = (
            " && ".join(
                f"module load {shlex.quote(m)} >/dev/null 2>&1" for m in modules
            )
            if modules
            else ""
        )
        path_prefix = (
            "export PATH="
            + ":".join(shlex.quote(p) for p in resolved_bin_dirs)
            + ":$PATH"
            if resolved_bin_dirs
            else ""
        )
        module_list = [str(m).strip() for m in modules_to_import if str(m).strip()]
        if not module_list:
            return
        check = (
            f"{shlex.quote(python_executable)} -c "
            + shlex.quote(
                "import importlib; "
                f"[importlib.import_module(m) for m in {module_list!r}]"
            )
        )
        parts = [p for p in [module_prefix, path_prefix, check] if p]
        wrapped = " && ".join(parts)
        rc, _, stderr = self._ssh.run(f"bash -lc {shlex.quote(wrapped)}", check=False)
        if rc != 0:
            raise RuntimeError(
                f"Remote Python import check failed for {python_executable!r} "
                f"modules {module_list}: {stderr.strip() or 'unknown error'}"
            )

    def _status_from_sacct(self, job_id: str) -> dict | None:
        """Return status details from sacct, or None when unavailable."""
        rc, stdout, _ = self._ssh.run(
            f"sacct -j {job_id} --format=JobID,State,ExitCode -n -P 2>/dev/null",
            check=False,
        )
        if rc != 0 or not stdout.strip():
            return None

        rows: list[tuple[str, str, str]] = []
        for line in stdout.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            rows.append((parts[0], parts[1], parts[2]))
        if not rows:
            return None

        selected = None
        for jid, state, exit_code in rows:
            if jid == job_id:
                selected = (jid, state, exit_code)
                break
        if selected is None:
            for jid, state, exit_code in rows:
                if "." not in jid:
                    selected = (jid, state, exit_code)
                    break
        if selected is None:
            selected = rows[0]

        _, raw_state, exit_code = selected
        state = raw_state.split()[0].upper()
        state = state.split("+", 1)[0]

        queued = {"PENDING", "CONFIGURING", "RESV_DEL_HOLD", "REQUEUE_HOLD", "QUEUED"}
        running = {"RUNNING", "COMPLETING", "SUSPENDED"}
        failed = {
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "PREEMPTED",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "BOOT_FAIL",
            "DEADLINE",
            "REVOKED",
        }

        exit_main = exit_code.split(":", 1)[0].strip() if exit_code else ""
        exit_ok = exit_main == "0"

        if state in queued:
            status: JobStatus = "QUEUED"
            terminal = False
        elif state in running:
            status = "RUNNING"
            terminal = False
        elif state == "COMPLETED" and exit_ok:
            status = "COMPLETED"
            terminal = True
        elif state in failed or (state == "COMPLETED" and not exit_ok):
            status = "FAILED"
            terminal = True
        else:
            status = "UNKNOWN"
            terminal = False

        return {
            "status": status,
            "scheduler_state": raw_state,
            "exit_code": exit_code,
            "source": "sacct",
            "terminal": terminal,
        }

    def _status_from_qstat(self, job_id: str) -> dict:
        """Return best-effort status details from qstat."""
        rc, stdout, _ = self._ssh.run(
            f"qstat -f {job_id} 2>/dev/null",
            check=False,
        )
        if rc != 0 or not stdout.strip():
            return {
                "status": "UNKNOWN",
                "scheduler_state": "NOTFOUND",
                "exit_code": None,
                "source": "qstat",
                "terminal": False,
            }

        for line in stdout.splitlines():
            if "job_state" in line.lower():
                state = line.split("=")[-1].strip().upper()
                if state in {"Q", "H", "W"}:
                    status: JobStatus = "QUEUED"
                    terminal = False
                elif state in {"R", "B", "T", "E"}:
                    status = "RUNNING"
                    terminal = False
                elif state == "C":
                    status = "COMPLETED"
                    terminal = True
                else:
                    status = "UNKNOWN"
                    terminal = False
                return {
                    "status": status,
                    "scheduler_state": state,
                    "exit_code": None,
                    "source": "qstat",
                    "terminal": terminal,
                }

        return {
            "status": "UNKNOWN",
            "scheduler_state": "UNPARSEABLE",
            "exit_code": None,
            "source": "qstat",
            "terminal": False,
        }

    def submit(
        self,
        script_content: str,
        remote_dir: str,
        *,
        script_name: str = "job.pbs",
    ) -> dict:
        """Upload and submit a PBS job script.

        Parameters
        ----------
        script_content : str
            PBS script content (from wtec.cluster.pbs.generate_script).
        remote_dir : str
            Remote working directory. Created if it doesn't exist.
        script_name : str
            Filename for the PBS script on the cluster.

        Returns
        -------
        dict
            Metadata containing parsed PBS job ID and remote script path.
        """
        import tempfile

        self._ssh.mkdir_p(remote_dir)

        # Write script to a temporary local file and upload
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pbs", delete=False
        ) as f:
            f.write(script_content)
            local_tmp = f.name

        try:
            remote_script = f"{remote_dir}/{script_name}"
            self._ssh.put(local_tmp, remote_script)
        finally:
            os.unlink(local_tmp)

        _, stdout, stderr = self._ssh.run(f"qsub {shlex.quote(remote_script)}")
        merged = (stdout + "\n" + stderr).strip()
        job_id = self._parse_qsub_job_id(merged)
        return {"job_id": job_id, "remote_script": remote_script}

    def status(self, job_id: str) -> JobStatus:
        """Query cluster job status with sacct-first resolution.

        Returns
        -------
        'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'UNKNOWN'
        """
        return self.status_details(job_id)["status"]

    def status_details(self, job_id: str) -> dict:
        """Return detailed status metadata for a job ID."""
        info = self._status_from_sacct(job_id)
        if info is not None:
            return info
        return self._status_from_qstat(job_id)

    def wait(
        self,
        job_id: str,
        *,
        poll_interval: int = 30,
        timeout: int = 86400,
        verbose: bool = True,
        live_log: bool = False,
        live_remote_dir: str | None = None,
        live_files: list[str] | None = None,
        stale_log_seconds: int = 300,
        stream_from_start: bool = False,
        cancel_event=None,
    ) -> dict:
        """Block until job reaches a terminal state.

        Parameters
        ----------
        poll_interval : int
            Seconds between qstat polls.
        timeout : int
            Maximum wait time in seconds (default 24h).

        Returns
        -------
        dict
            Final status metadata dict from status_details().
        """
        elapsed = 0
        stale_warned = False
        last_log_growth = time.time()
        log_offsets: dict[str, int | None] = {}
        log_paths: list[str] = []
        cancel_requested = False
        if live_log and live_remote_dir and live_files:
            for name in live_files:
                if not name:
                    continue
                if name.startswith("/"):
                    log_paths.append(name)
                else:
                    log_paths.append(posixpath.join(live_remote_dir, name))
            if verbose and log_paths:
                print("  Live log watch:")
                for rp in log_paths:
                    print(f"    - {rp}")
        while elapsed < timeout:
            details = self.status_details(job_id)
            s = details["status"]
            had_growth = False
            if live_log and log_paths:
                for remote_path in log_paths:
                    grew, off = self._stream_file_delta(
                        remote_path,
                        log_offsets.get(remote_path),
                        stream_from_start=stream_from_start,
                    )
                    log_offsets[remote_path] = off
                    had_growth = had_growth or grew
            if had_growth:
                last_log_growth = time.time()
                stale_warned = False
            if verbose:
                state = details.get("scheduler_state")
                src = details.get("source")
                print(
                    f"  Job {job_id}: {s} [{state}] via {src} ({elapsed}s elapsed)"
                )
            if (
                cancel_event is not None
                and hasattr(cancel_event, "is_set")
                and cancel_event.is_set()
                and not details.get("terminal")
                and not cancel_requested
            ):
                cancel_requested = True
                self._ssh.run(f"qdel {shlex.quote(job_id)} 2>/dev/null", check=False)
                if verbose:
                    print(f"  Cancel requested for job {job_id}; issued qdel")
            if (
                live_log
                and s == "RUNNING"
                and stale_log_seconds > 0
                and not stale_warned
                and (time.time() - last_log_growth) >= stale_log_seconds
            ):
                stale_warned = True
                print(
                    "  WARNING: scheduler reports RUNNING but watched logs "
                    f"have not grown for ~{stale_log_seconds}s."
                )
                print(f"  Suggestion: wtec status --job-id {job_id}")
                print(f"  If confirmed stale: qdel {job_id}")
            if details.get("terminal"):
                return details
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

    def retrieve(
        self,
        remote_dir: str,
        local_dir: str | Path,
        patterns: list[str],
    ) -> None:
        """Download output files from cluster.

        Parameters
        ----------
        patterns : list[str]
            Filename glob patterns, e.g. ['*_hr.dat', '*.wout', 'scf.out'].
        """
        self._ssh.get_many(remote_dir, local_dir, patterns)

    def submit_and_wait(
        self,
        script_content: str,
        remote_dir: str,
        local_dir: str | Path,
        retrieve_patterns: list[str],
        *,
        script_name: str = "job.pbs",
        stage_files: list[str | Path] | None = None,
        expected_local_outputs: list[str | Path] | None = None,
        queue_used: str | None = None,
        poll_interval: int = 30,
        verbose: bool = True,
        live_log: bool = False,
        live_files: list[str] | None = None,
        stale_log_seconds: int = 300,
        retrieve_on_failure: bool = False,
        stream_from_start: bool = False,
        cancel_event=None,
    ) -> dict:
        """Submit job, wait for completion, retrieve files.

        Returns
        -------
        dict with terminal status metadata and output location
        """
        if stage_files:
            staged = self.stage_files(stage_files, remote_dir)
            if verbose:
                print(f"  Staged {len(staged)} input file(s) to {remote_dir}")

        submitted = self.submit(script_content, remote_dir, script_name=script_name)
        job_id = submitted["job_id"]
        if verbose:
            print(f"  Submitted job {job_id} in {remote_dir}")

        final_details = self.wait(
            job_id,
            poll_interval=poll_interval,
            verbose=verbose,
            live_log=live_log,
            live_remote_dir=remote_dir,
            live_files=live_files,
            stale_log_seconds=stale_log_seconds,
            stream_from_start=stream_from_start,
            cancel_event=cancel_event,
        )
        final_status = final_details["status"]

        should_retrieve = final_status not in ("FAILED", "UNKNOWN") or bool(retrieve_on_failure)
        retrieve_error: str | None = None
        if should_retrieve:
            try:
                self.retrieve(remote_dir, local_dir, retrieve_patterns)
            except Exception as exc:
                retrieve_error = f"{type(exc).__name__}: {exc}"

        if expected_local_outputs and final_status not in ("FAILED", "UNKNOWN"):
            missing_or_empty: list[str] = []
            local_dir_path = Path(local_dir)
            for expected in expected_local_outputs:
                p = Path(expected)
                if not p.is_absolute():
                    p = local_dir_path / p
                if not p.exists() or p.stat().st_size == 0:
                    missing_or_empty.append(str(p))
            if missing_or_empty:
                raise RuntimeError(
                    "Missing/empty expected local output file(s):\n  "
                    + "\n  ".join(missing_or_empty)
                )

        if verbose:
            if should_retrieve and retrieve_error is None:
                print(f"  Retrieved files to {local_dir}")
            elif should_retrieve and retrieve_error is not None:
                print(f"  WARNING: retrieval failed for {local_dir}: {retrieve_error}")

        if final_status in ("FAILED", "UNKNOWN"):
            retrieve_note = ""
            if should_retrieve:
                if retrieve_error is None:
                    retrieve_note = " Retrieved requested artifacts locally."
                else:
                    retrieve_note = f" Artifact retrieval failed: {retrieve_error}."
            raise RuntimeError(
                "Job "
                f"{job_id} ended with status={final_status} "
                f"(scheduler_state={final_details.get('scheduler_state')}, "
                f"exit_code={final_details.get('exit_code')}, source={final_details.get('source')}). "
                f"Check {remote_dir} for logs and outputs.{retrieve_note}"
            )

        return {
            "job_id": job_id,
            "status": final_status,
            "scheduler_state": final_details.get("scheduler_state"),
            "exit_code": final_details.get("exit_code"),
            "status_source": final_details.get("source"),
            "queue": queue_used,
            "remote_dir": remote_dir,
            "remote_script": submitted.get("remote_script"),
            "local_dir": str(local_dir),
        }

    # ---- live-log helpers -------------------------------------------------

    def _remote_file_size(self, remote_path: str) -> int | None:
        cmd = (
            f"if [ -f {shlex.quote(remote_path)} ]; then "
            f"wc -c < {shlex.quote(remote_path)}; "
            "else echo -1; fi"
        )
        rc, out, _ = self._ssh.run(cmd, check=False)
        if rc != 0:
            return None
        raw = out.strip().splitlines()
        if not raw:
            return None
        try:
            size = int(raw[-1].strip())
        except Exception:
            return None
        if size < 0:
            return None
        return size

    def _remote_read_from(self, remote_path: str, start_byte_1: int) -> str:
        start = max(1, int(start_byte_1))
        cmd = f"tail -c +{start} {shlex.quote(remote_path)} 2>/dev/null || true"
        rc, out, _ = self._ssh.run(cmd, check=False)
        if rc != 0:
            return ""
        return out

    def _stream_file_delta(
        self,
        remote_path: str,
        current_offset: int | None,
        *,
        stream_from_start: bool = False,
    ) -> tuple[bool, int | None]:
        """Stream newly appended bytes from remote file.

        Returns
        -------
        (had_growth, new_offset)
        """
        size = self._remote_file_size(remote_path)
        if size is None:
            return (False, current_offset)

        # First observation either starts at file end (default) or from byte 1.
        if current_offset is None:
            if not stream_from_start:
                return (False, size)
            offset = 0
            if size == 0:
                return (False, offset)
            delta0 = self._remote_read_from(remote_path, 1)
            if delta0:
                tag = posixpath.basename(remote_path)
                for line in delta0.splitlines():
                    print(f"  [{tag}] {line}")
            return (bool(delta0), size)

        offset = current_offset
        if size < offset:
            # File rotated or truncated; restart from beginning.
            offset = 0

        if size == offset:
            return (False, offset)

        delta = self._remote_read_from(remote_path, offset + 1)
        if delta:
            tag = posixpath.basename(remote_path)
            for line in delta.splitlines():
                print(f"  [{tag}] {line}")
        return (True, size)
