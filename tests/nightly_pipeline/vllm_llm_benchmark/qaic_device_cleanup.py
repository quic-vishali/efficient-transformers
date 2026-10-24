#!/usr/bin/env python3
"""Release only QAIC server process groups owned by this Jenkins run.

This helper intentionally does not allocate fallback devices and does not kill
arbitrary processes.  It performs three actions:

1. Query ``qaic-util -q`` for the requested QIDs.
2. If a requested QID is busy, find server processes using that device group.
3. Terminate a process group only when its ownership matches this run.

The benchmark runner starts servers with ``setsid``; therefore each server has
its own process group and can be terminated without touching other jobs.
Ownership is proved using ``JENKINS_RUN_ID``, ``BUILD_TAG``, or
``JENKINS_SERVER_COOKIE``.  When running inside the per-build Docker container,
the container hostname is used as a fallback scope.  If no ownership scope can
be proved, this module refuses to kill anything.

Use ``--force-kill`` only when intentionally cleaning up another process.
Force mode bypasses ownership checks, but still requires the process to expose
at least one QID requested by this cleanup. It relies on normal OS permissions
to terminate the process group.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


class DeviceCleanupError(RuntimeError):
    """Raised when device ownership cannot be checked safely."""


@dataclass(frozen=True)
class DeviceState:
    qid: int
    status: str
    nsp_free: int | None
    nsp_total: int | None

    @property
    def ready(self) -> bool:
        return self.status.lower().startswith("ready")

    @property
    def busy(self) -> bool:
        return (
            self.ready
            and self.nsp_free is not None
            and self.nsp_total is not None
            and self.nsp_total > 0
            and self.nsp_free < self.nsp_total
        )

    @property
    def parseable(self) -> bool:
        """Whether qaic-util returned enough resource data to make a decision."""
        return (
            self.status.lower() != "unknown"
            and self.nsp_free is not None
            and self.nsp_total is not None
            and self.nsp_total > 0
            and 0 <= self.nsp_free <= self.nsp_total
        )


@dataclass(frozen=True)
class OwnedProcess:
    pid: int
    pgid: int
    command: tuple[str, ...]
    matched_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CleanupResult:
    busy_qids: tuple[int, ...]
    matched_processes: tuple[OwnedProcess, ...]
    killed_process_groups: tuple[int, ...]
    ownership_scope: str
    # Processes that used a requested QID but were excluded by
    # Jenkins/UID ownership checks. This is diagnostic-only and is populated
    # when a non-force cleanup finds no owned match.
    other_matching_processes: tuple[OwnedProcess, ...] = ()


_QID_HEADER_RE = re.compile(r"(?im)^\s*QID\s*(?::|=)?\s*(\d+)\s*$")
_STATUS_RE = re.compile(r"(?im)^\s*Status\s*(?::|=)\s*(.+?)\s*$")
_NSP_TOTAL_RE = re.compile(r"(?im)^\s*Nsp\s+Total\s*(?::|=)\s*(\d+)")
_NSP_FREE_RE = re.compile(r"(?im)^\s*Nsp\s+Free\s*(?::|=)\s*(\d+)")
_GROUP_FLAGS = {
    "--device-group",
    "--encode-device-group",
    "--prefill-device-group",
    "--decode-device-group",
}
_GROUP_ENV_NAMES = {
    "QAIC_DEVICE_IDS",
    "QAIC_DEVICE_GROUP",
    # vLLM QAIC workers may inherit the selected group through this variable
    # after platform initialization, while their child command line no longer
    # contains --additional-config.
    "QAIC_VISIBLE_DEVICES",
    "DEVICE_GROUP",
    "ENCODE_DEVICE_GROUP",
    "PREFILL_DEVICE_GROUP",
    "DECODE_DEVICE_GROUP",
}
_RUN_MARKER_NAMES = ("JENKINS_RUN_ID", "BUILD_TAG", "JENKINS_SERVER_COOKIE")


def parse_group(raw: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        normalized = raw.strip()
        if normalized.startswith("[") and normalized.endswith("]"):
            normalized = normalized[1:-1].strip()
        values = [item for item in re.split(r"[,\s:]+", normalized) if item]
        if not values:
            raise ValueError(f"empty device group: {raw!r}")
        group = tuple(int(item) for item in values)
    else:
        group = tuple(int(item) for item in raw)
    if not group or any(qid < 0 for qid in group):
        raise ValueError(f"invalid device group: {raw!r}")
    if len(set(group)) != len(group):
        raise ValueError(f"duplicate QID in device group: {raw!r}")
    return group


def parse_groups(raw: str | Sequence[Sequence[int] | str]) -> tuple[tuple[int, ...], ...]:
    if isinstance(raw, str):
        groups = tuple(parse_group(item) for item in raw.split(";") if item.strip())
    else:
        groups = tuple(parse_group(item) for item in raw)
    if not groups:
        raise ValueError("at least one device group is required")
    flattened = [qid for group in groups for qid in group]
    if len(set(flattened)) != len(flattened):
        raise ValueError("device groups must be disjoint")
    return groups


def _parse_qaic_util_output(output: str, qids: Iterable[int]) -> dict[int, DeviceState]:
    headers = list(_QID_HEADER_RE.finditer(output))
    blocks: dict[int, str] = {}
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(output)
        blocks[int(header.group(1))] = output[header.start() : end]

    states = {}
    for qid in qids:
        block = blocks.get(qid, "")
        status_match = _STATUS_RE.search(block)
        total_match = _NSP_TOTAL_RE.search(block)
        free_match = _NSP_FREE_RE.search(block)
        states[qid] = DeviceState(
            qid=qid,
            status=status_match.group(1).strip() if status_match else "Unknown",
            nsp_free=int(free_match.group(1)) if free_match else None,
            nsp_total=int(total_match.group(1)) if total_match else None,
        )
    return states


def query_device_states(
    qids: Iterable[int],
    *,
    qaic_util: str = "/opt/qti-aic/tools/qaic-util",
) -> dict[int, DeviceState]:
    requested = tuple(sorted(set(int(qid) for qid in qids)))
    if not requested:
        raise ValueError("at least one QID is required")
    try:
        result = subprocess.run(
            [qaic_util, "-q"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        raise DeviceCleanupError(f"cannot run {qaic_util} -q: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise DeviceCleanupError(f"{qaic_util} -q failed: {detail}")
    return _parse_qaic_util_output(result.stdout, requested)


def _read_proc_file(pid: int, name: str) -> bytes | None:
    try:
        return Path(f"/proc/{pid}/{name}").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _process_environment(pid: int) -> dict[str, str]:
    raw = _read_proc_file(pid, "environ")
    if raw is None:
        return {}
    environment = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        try:
            environment[key.decode()] = value.decode(errors="replace")
        except UnicodeDecodeError:
            continue
    return environment


def _process_command(pid: int) -> tuple[str, ...]:
    raw = _read_proc_file(pid, "cmdline")
    if not raw:
        return ()
    return tuple(item.decode(errors="replace") for item in raw.split(b"\0") if item)


def _current_ownership_scope() -> tuple[str, str] | None:
    for name in _RUN_MARKER_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return name, value

    # Jenkins creates one Docker container per build in this pipeline.  Only
    # use HOSTNAME as a fallback inside Docker; using the host machine's
    # hostname could incorrectly match another Jenkins job.
    if Path("/.dockerenv").exists():
        hostname = os.environ.get("HOSTNAME", "").strip()
        if hostname:
            return "HOSTNAME", hostname
    return None


def _belongs_to_scope(environment: dict[str, str], scope: tuple[str, str]) -> bool:
    name, value = scope
    return environment.get(name, "") == value


def _extract_groups(command: Sequence[str], environment: dict[str, str]) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token == "--additional-config" and index + 1 < len(command):
            try:
                additional_config = json.loads(command[index + 1])
                if not isinstance(additional_config, dict):
                    raise ValueError("additional-config must be a JSON object")
                device_group = additional_config.get("device_group")
                if device_group is not None:
                    groups.append(parse_group(device_group))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            index += 2
            continue
        flag, equals, inline = token.partition("=")
        if flag not in _GROUP_FLAGS:
            index += 1
            continue
        values = [inline] if equals else []
        index += 1
        while not equals and index < len(command) and not command[index].startswith("--"):
            values.append(command[index])
            index += 1
        try:
            group = parse_group(",".join(values))
        except ValueError:
            continue
        groups.append(group)

    for name in _GROUP_ENV_NAMES:
        raw = environment.get(name, "").strip()
        if not raw:
            continue
        try:
            groups.append(parse_group(raw))
        except ValueError:
            continue
    return tuple(groups)


def extract_device_groups(
    command: Sequence[str],
    environment: dict[str, str] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Extract device groups from a server command and optional environment."""
    return _extract_groups(command, environment or {})


def _process_matches_groups(
    command: Sequence[str],
    environment: dict[str, str],
    requested_groups: tuple[tuple[int, ...], ...],
    busy_qids: set[int],
) -> tuple[tuple[int, ...], ...]:
    process_groups = _extract_groups(command, environment)
    matched = []
    for process_group in process_groups:
        process_qids = set(process_group)
        for requested in requested_groups:
            if process_qids.intersection(requested).intersection(busy_qids):
                # Report the group exposed by the process, not the requested
                # group. This matters when overlap matching accepts a process
                # using (0,) for a request covering (0, 1).
                if process_group not in matched:
                    matched.append(process_group)
                break
    return tuple(matched)


def find_owned_processes(
    requested_groups: tuple[tuple[int, ...], ...],
    busy_qids: set[int],
    scope: tuple[str, str] | None,
    same_user_only: bool = True,
) -> tuple[OwnedProcess, ...]:
    owned = []
    current_pgid = os.getpgid(os.getpid())
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            if same_user_only and entry.stat().st_uid != os.getuid():
                continue
            pgid = os.getpgid(pid)
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if pgid <= 1 or pgid == current_pgid:
            continue
        environment = _process_environment(pid)
        if scope is not None and not _belongs_to_scope(environment, scope):
            continue
        command = _process_command(pid)
        matched_groups = _process_matches_groups(command, environment, requested_groups, busy_qids)
        if matched_groups:
            owned.append(OwnedProcess(pid, pgid, command, matched_groups))
    return tuple(owned)


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _terminate_groups(pgids: Iterable[int], timeout_s: float) -> tuple[int, ...]:
    unique_pgids = tuple(sorted(set(pgids)))
    signaled_pgids = set()
    for pgid in unique_pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
            signaled_pgids.add(pgid)
        except (ProcessLookupError, PermissionError, OSError):
            continue

    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline and any(_process_group_exists(pgid) for pgid in unique_pgids):
        time.sleep(0.1)

    for pgid in unique_pgids:
        if _process_group_exists(pgid):
            try:
                os.killpg(pgid, signal.SIGKILL)
                signaled_pgids.add(pgid)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    return tuple(sorted(signaled_pgids))


def _wait_for_devices_ready(
    qids: Iterable[int],
    *,
    qaic_util: str,
    timeout_s: float,
) -> None:
    """Wait until QAIC reports every released QID ready and fully free."""
    requested = tuple(sorted(set(qids)))
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        states = query_device_states(requested, qaic_util=qaic_util)
        _validate_device_states(states)
        if all(
            state.ready
            and state.nsp_free is not None
            and state.nsp_total is not None
            and state.nsp_free == state.nsp_total
            for state in states.values()
        ):
            return
        if time.monotonic() >= deadline:
            raise DeviceCleanupError(
                "QAIC devices remain unavailable after process cleanup: " + ",".join(map(str, requested))
            )
        time.sleep(0.5)


def _validate_device_states(states: dict[int, DeviceState]) -> None:
    invalid = {qid: state for qid, state in states.items() if not state.parseable}
    if invalid:
        details = "; ".join(
            f"{qid}: status={state.status!r}, nsp_free={state.nsp_free!r}, nsp_total={state.nsp_total!r}"
            for qid, state in sorted(invalid.items())
        )
        raise DeviceCleanupError(f"could not determine QAIC state: {details}")


def release_owned_busy_processes(
    requested_groups: str | Sequence[Sequence[int] | str],
    *,
    qaic_util: str = "/opt/qti-aic/tools/qaic-util",
    terminate_timeout_s: float = 10.0,
    dry_run: bool = False,
    force_kill: bool = False,
) -> CleanupResult:
    groups = parse_groups(requested_groups)
    qids = {qid for group in groups for qid in group}
    states = query_device_states(qids, qaic_util=qaic_util)
    _validate_device_states(states)
    busy_qids = {qid for qid, state in states.items() if state.busy}
    scope = _current_ownership_scope()
    if not busy_qids:
        return CleanupResult((), (), (), f"{scope[0]}={scope[1]}" if scope else "none")
    if scope is None and not force_kill:
        raise DeviceCleanupError("requested QIDs are busy, but Jenkins ownership cannot be proved; refusing to kill")

    owned = find_owned_processes(
        groups,
        busy_qids,
        None if force_kill else scope,
        same_user_only=not force_kill,
    )
    other_matching = ()
    if not force_kill:
        # Diagnostic scan only: do not use these processes for termination.
        # This distinguishes ownership mismatches from processes that do not
        # expose any requested busy QID. Keep only processes not already in
        # the ownership-filtered result.
        all_matching = find_owned_processes(
            groups,
            busy_qids,
            None,
            same_user_only=False,
        )
        owned_keys = {(process.pid, process.pgid) for process in owned}
        other_matching = tuple(process for process in all_matching if (process.pid, process.pgid) not in owned_keys)
    pgids = tuple(process.pgid for process in owned)
    killed = () if dry_run else _terminate_groups(pgids, terminate_timeout_s)
    if killed:
        _wait_for_devices_ready(
            qids,
            qaic_util=qaic_util,
            timeout_s=terminate_timeout_s,
        )
    return CleanupResult(
        busy_qids=tuple(sorted(busy_qids)),
        matched_processes=owned,
        killed_process_groups=killed,
        ownership_scope=("FORCE_KILL: ownership checks bypassed" if force_kill else f"{scope[0]}={scope[1]}"),
        other_matching_processes=other_matching,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", required=True, help="Requested groups, e.g. '0,1,2,3' or '0,1;2,3'")
    parser.add_argument("--qaic-util", default="/opt/qti-aic/tools/qaic-util")
    parser.add_argument("--terminate-timeout-s", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force-kill",
        action="store_true",
        help=("Bypass Jenkins/UID ownership checks when a process uses any requested QID; OS permission is required"),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = release_owned_busy_processes(
            args.groups,
            qaic_util=args.qaic_util,
            terminate_timeout_s=args.terminate_timeout_s,
            dry_run=args.dry_run,
            force_kill=args.force_kill,
        )
    except (DeviceCleanupError, OSError, ValueError) as exc:
        print(f"device cleanup failed: {exc}")
        return 2
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
