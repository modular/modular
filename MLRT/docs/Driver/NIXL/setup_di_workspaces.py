#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Set up two AWS B200 Coder workspaces for disaggregated-inference EFA testing.

Configures the SSH keys, ``/etc/hosts`` entries, and known_hosts state needed
between a prefill node and a decode node so that ``mpirun``, the libfabric
backend, and the disaggregated-inference dispatcher can talk both ways.

Usage:
    python3 setup_di_workspaces.py <prefill-workspace> <decode-workspace>

Example:
    python3 setup_di_workspaces.py josh-linux-b200-aws-dev josh-linux-b200-aws-dev-2

After it finishes, follow ``run_efa_di_benchmark.md`` next to this script to
launch ``run_dist`` and run a benchmark.

The script is idempotent: re-running it on an already-configured pair of
workspaces is a no-op aside from the verification at the end.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import textwrap


class C:
    """ANSI escape codes for terminal styling."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    BRIGHT_CYAN = "\033[96m"


USE_COLOR = sys.stdout.isatty()


def style(text: str, *codes: str) -> str:
    if not USE_COLOR:
        return text
    return "".join(codes) + text + C.RESET


def banner(text: str) -> None:
    bar = "━" * (len(text) + 4)
    print()
    print(style(bar, C.MAGENTA, C.BOLD))
    print(style(f"  {text}  ", C.MAGENTA, C.BOLD))
    print(style(bar, C.MAGENTA, C.BOLD))


def step(text: str) -> None:
    print(style("▸ ", C.CYAN, C.BOLD) + style(text, C.BOLD))


def ok(text: str) -> None:
    print(style("  ✓ ", C.GREEN, C.BOLD) + text)


def warn(text: str) -> None:
    print(style("  ! ", C.YELLOW, C.BOLD) + text)


def err(text: str) -> None:
    print(style("  ✗ ", C.RED, C.BOLD) + text)


def hint(text: str) -> None:
    print(style(f"    {text}", C.DIM))


def ask(question: str) -> str:
    return input(style(f"? {question} ", C.YELLOW, C.BOLD)).strip()


def fatal(text: str) -> None:
    err(text)
    sys.exit(1)


def run(cmd: list[str], *, check: bool = True) -> tuple[str, str, int]:
    """Run a local command. Returns ``(stdout, stderr, returncode)``.

    Raises ``SystemExit`` on a non-zero exit when ``check=True``.
    """
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as e:
        fatal(f"Command not found: {e.filename}. Is it on PATH?")
    if check and result.returncode != 0:
        err(
            f"Command failed (exit {result.returncode}): {' '.join(shlex.quote(c) for c in cmd)}"
        )
        if result.stderr:
            hint(result.stderr.strip())
        sys.exit(1)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def coder_ssh(
    workspace: str, remote_command: str, *, check: bool = True
) -> tuple[str, str, int]:
    """Run a shell command on a Coder workspace via ``coder ssh``."""
    return run(["coder", "ssh", workspace, "--", remote_command], check=check)


def get_coder_owner() -> str:
    """Return the logged-in Coder username, or a placeholder on failure.

    Runs ``coder whoami --output json`` and extracts the ``username`` field.
    Falls back to the literal string ``<your-coder-username>`` if the command
    fails or the field is not found.
    """
    stdout, _, rc = run(["coder", "whoami", "--output", "json"], check=False)
    if rc != 0:
        return "<your-coder-username>"
    try:
        data = json.loads(stdout)
        # coder whoami -o json returns a list with one entry
        if isinstance(data, list) and data:
            return data[0].get("username", "<your-coder-username>")
        if isinstance(data, dict):
            return data.get("username", "<your-coder-username>")
    except (json.JSONDecodeError, KeyError):
        pass
    return "<your-coder-username>"


def coder_status(workspace: str) -> str | None:
    """Return the latest-build status of a workspace, or ``None`` if not found."""
    stdout, _, _ = run(["coder", "list", "-o", "json"])
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        fatal(
            "`coder list -o json` did not return valid JSON. Is the Coder CLI logged in?"
        )
    for ws in data:
        if ws.get("name") == workspace:
            return ws.get("latest_build", {}).get("status")
    return None


def ensure_running(workspace: str) -> None:
    step(f"Checking workspace {style(workspace, C.BOLD, C.BRIGHT_CYAN)}")
    status = coder_status(workspace)
    if status is None:
        fatal(
            f"Workspace '{workspace}' not found. Check spelling or run `coder list`."
        )
    if status == "running":
        ok("Already running")
        return
    warn(f"Status is '{status}', not running")
    answer = ask(f"Start it now with `coder start -y {workspace}`? [y/N]")
    if answer.lower() not in ("y", "yes"):
        fatal("Cannot proceed with a stopped workspace.")
    hint("Starting workspace (this can take ~1 minute)…")
    run(["coder", "start", "-y", workspace], check=True)
    status = coder_status(workspace)
    if status != "running":
        fatal(f"Workspace did not reach running state (status: {status})")
    ok("Started")


def get_ipv6(workspace: str) -> str:
    """Return the eth0 global-scope IPv6 address."""
    cmd = "ip -6 addr show eth0 | awk '/scope global/ {print $2}' | head -1 | cut -d/ -f1"
    stdout, _, _ = coder_ssh(workspace, cmd)
    if not stdout:
        fatal(f"Could not determine global IPv6 for {workspace}")
    return stdout


def get_hostname(workspace: str) -> str:
    stdout, _, _ = coder_ssh(workspace, "hostname")
    if not stdout:
        fatal(f"Could not determine hostname for {workspace}")
    return stdout


def ensure_ssh_key(workspace: str) -> str:
    """Make sure ``~/.ssh/id_ed25519`` exists, return the public key."""
    coder_ssh(
        workspace,
        '[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -q',
    )
    stdout, _, _ = coder_ssh(workspace, "cat ~/.ssh/id_ed25519.pub")
    if not stdout:
        fatal(f"Could not read public key on {workspace}")
    return stdout


def add_authorized_key(workspace: str, pub_key: str, peer_label: str) -> None:
    """Append ``pub_key`` to authorized_keys on workspace if not already present."""
    # Match against the base64 portion (the second whitespace-separated field)
    # so a whitespace-mangled paste still detects an existing entry.
    parts = pub_key.split()
    if len(parts) < 2:
        fatal(f"Public key from {peer_label} looks malformed: {pub_key[:60]}…")
    fingerprint = parts[1]
    check = (
        f"grep -qF {shlex.quote(fingerprint)} ~/.ssh/authorized_keys 2>/dev/null "
        f"&& echo present || echo absent"
    )
    stdout, _, _ = coder_ssh(workspace, check)
    if "present" in stdout:
        ok(f"{peer_label}'s key already authorized on {workspace}")
        return
    add_cmd = (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
        f"echo {shlex.quote(pub_key)} >> ~/.ssh/authorized_keys && "
        "chmod 600 ~/.ssh/authorized_keys"
    )
    coder_ssh(workspace, add_cmd)
    ok(f"Authorized {peer_label}'s key on {workspace}")


def ensure_etc_hosts(workspace: str, ipv6: str, name: str, label: str) -> None:
    """Ensure ``ipv6 name`` is resolvable on ``workspace``."""
    stdout, _, _ = coder_ssh(
        workspace, f"getent hosts {shlex.quote(name)}", check=False
    )
    if stdout and ipv6 in stdout:
        ok(f"{label} '{name}' already in /etc/hosts on {workspace}")
        return
    if stdout:
        warn(
            f"/etc/hosts on {workspace} resolves '{name}' to {stdout.split()[0]}"
        )
        warn(f"Expected: {ipv6}")
        answer = ask("Replace the stale entry? [y/N]")
        if answer.lower() in ("y", "yes"):
            sed_cmd = (
                f"sudo sed -i.bak '/[[:space:]]{name}\\([[:space:]]\\|$\\)/d' /etc/hosts "
                f"&& echo {shlex.quote(f'{ipv6} {name}')} | sudo tee -a /etc/hosts >/dev/null"
            )
            coder_ssh(workspace, sed_cmd)
            ok(f"Replaced {label} entry for '{name}' on {workspace}")
        else:
            warn("Skipped; transfer may fail until /etc/hosts is fixed.")
        return
    add_cmd = f"echo {shlex.quote(f'{ipv6} {name}')} | sudo tee -a /etc/hosts >/dev/null"
    coder_ssh(workspace, add_cmd)
    ok(f"Added {label} entry '{ipv6} {name}' to /etc/hosts on {workspace}")


def ensure_ssh_config_alias(workspace: str, alias: str) -> None:
    """Ensure ``~/.ssh/config`` accepts new host keys for the alias non-interactively.

    mpirun and ssh-from-the-other-pod both run in batch mode; without
    ``StrictHostKeyChecking=accept-new`` for our short-lived pod aliases the
    first connection from a fresh pod fails with "Host key verification failed".
    """
    block = textwrap.dedent(
        f"""
        # Added by setup_di_workspaces.py for EFA DI experiments
        Host {alias}
            StrictHostKeyChecking accept-new
        """
    ).strip()
    check = (
        "test -f ~/.ssh/config && "
        f"grep -qE '^Host [^#]*\\b{alias}\\b' ~/.ssh/config "
        "&& echo present || echo absent"
    )
    stdout, _, _ = coder_ssh(workspace, check)
    if "present" in stdout:
        ok(f"~/.ssh/config on {workspace} already has '{alias}' block")
        return
    add_cmd = (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/config && "
        "chmod 600 ~/.ssh/config && "
        f"printf '\\n%s\\n' {shlex.quote(block)} >> ~/.ssh/config"
    )
    coder_ssh(workspace, add_cmd)
    ok(f"Added '{alias}' block to ~/.ssh/config on {workspace}")


def test_ssh(from_workspace: str, to_address: str, label: str) -> bool:
    cmd = (
        f"ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
        f"-o ConnectTimeout=5 {shlex.quote(to_address)} hostname"
    )
    stdout, stderr, rc = coder_ssh(from_workspace, cmd, check=False)
    if rc == 0 and stdout:
        ok(f"SSH {from_workspace} → {label} works (peer says: {stdout})")
        return True
    err(f"SSH {from_workspace} → {label} failed (rc={rc})")
    if stderr:
        hint(stderr)
    return False


def ensure_modular_clone(workspace: str) -> None:
    """Clone the Modular monorepo to ``~/code/modular`` if not already there.

    The DI launcher and benchmark commands all run from inside that working
    directory; without it, every later step is a non-starter. We try the SSH
    remote first (the workspace template ships a GitHub-registered key for
    `git@github.com:modularml/modular`), then fall back to HTTPS via the
    `gh` CLI's git credential helper if SSH fails.
    """
    step(f"Checking ~/code/modular on {workspace}")
    check_cmd = (
        "[ -d ~/code/modular/.git ] && echo present "
        "|| { [ -e ~/code/modular ] && echo exists-not-git "
        "|| echo absent; }"
    )
    stdout, _, _ = coder_ssh(workspace, check_cmd)
    if "present" in stdout:
        ok(f"~/code/modular is already a git checkout on {workspace}")
        return
    if "exists-not-git" in stdout:
        warn(f"~/code/modular exists on {workspace} but is not a git checkout")
        answer = ask("Skip cloning and trust whatever's there? [Y/n]")
        if not answer or answer.lower() in ("y", "yes"):
            return
        fatal(
            "Refusing to clobber an existing non-git ~/code/modular. "
            "Move or remove it manually, then rerun this script."
        )

    warn(f"~/code/modular not found on {workspace}; cloning from GitHub")
    hint("This typically takes 1-3 minutes for a fresh clone.")
    clone_cmd = (
        "mkdir -p ~/code && cd ~/code && "
        "git clone git@github.com:modularml/modular.git modular"
    )
    _, stderr, rc = coder_ssh(workspace, clone_cmd, check=False)
    if rc != 0:
        warn("SSH clone failed; trying HTTPS via gh-auth credential helper")
        hint(stderr.splitlines()[-1] if stderr else "(no stderr)")
        clone_https_cmd = (
            "mkdir -p ~/code && cd ~/code && "
            "git clone https://github.com/modularml/modular.git modular"
        )
        _, stderr2, rc2 = coder_ssh(workspace, clone_https_cmd, check=False)
        if rc2 != 0:
            err("Both SSH and HTTPS clone attempts failed.")
            hint(stderr2.splitlines()[-1] if stderr2 else "(no stderr)")
            hint(
                "On the workspace, set up GitHub access -- either an SSH key "
                "registered at https://github.com/settings/keys, or "
                "`gh auth login` for HTTPS -- then rerun this script."
            )
            sys.exit(1)
    ok(f"Cloned modular repo to ~/code/modular on {workspace}")
    step(f"Initializing submodules on {workspace}")
    _, _, _ = coder_ssh(
        workspace,
        "cd ~/code/modular && git submodule update --init --recursive",
    )
    ok(f"Submodules initialized on {workspace}")


def gpu_check(workspace: str) -> None:
    """Warn if the workspace's GPUs already have processes attached."""
    cmd = "nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader"
    stdout, _, rc = coder_ssh(workspace, cmd, check=False)
    if rc != 0:
        warn(f"Could not query GPU state on {workspace} (no nvidia-smi?)")
        return
    if not stdout:
        ok(f"No GPU consumers on {workspace}")
        return
    warn(f"{workspace} has existing GPU consumers:")
    for line in stdout.splitlines():
        hint(line)
    hint(
        "If those are leftover from a previous crashed run, kill them with "
        "`sudo kill -KILL <pid>` (defunct/zombie processes need root to be reaped)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prefill",
        help="Coder workspace name for the prefill node "
        "(e.g. josh-linux-b200-aws-dev)",
    )
    parser.add_argument(
        "decode",
        help="Coder workspace name for the decode node "
        "(e.g. josh-linux-b200-aws-dev-2)",
    )
    args = parser.parse_args()

    if args.prefill == args.decode:
        fatal("prefill and decode must be different workspaces.")

    banner("✨ EFA disaggregated-inference workspace setup ✨")
    print(f"  Prefill node: {style(args.prefill, C.BOLD, C.GREEN)}")
    print(f"  Decode node:  {style(args.decode, C.BOLD, C.GREEN)}")
    print()
    print(
        style(
            "  This script will SSH to both workspaces, clone the Modular "
            "repo if\n  missing, generate ed25519 keys if missing and "
            "cross-install them,\n  add /etc/hosts entries for each peer, "
            "and verify bidirectional\n  non-interactive SSH works.",
            C.DIM,
        )
    )
    print()
    answer = ask("Proceed? [Y/n]")
    if answer and answer.lower() not in ("y", "yes"):
        fatal("Aborted.")

    banner("Step 1 / 7 · Verify both workspaces are running")
    ensure_running(args.prefill)
    ensure_running(args.decode)

    banner("Step 2 / 7 · Modular monorepo")
    ensure_modular_clone(args.prefill)
    ensure_modular_clone(args.decode)

    banner("Step 3 / 7 · Gather network info")
    step(f"Querying {args.prefill}")
    p_ipv6 = get_ipv6(args.prefill)
    p_host = get_hostname(args.prefill)
    ok(f"IPv6:     {style(p_ipv6, C.BRIGHT_CYAN)}")
    ok(f"Hostname: {style(p_host, C.BRIGHT_CYAN)}")
    step(f"Querying {args.decode}")
    d_ipv6 = get_ipv6(args.decode)
    d_host = get_hostname(args.decode)
    ok(f"IPv6:     {style(d_ipv6, C.BRIGHT_CYAN)}")
    ok(f"Hostname: {style(d_host, C.BRIGHT_CYAN)}")

    banner("Step 4 / 7 · SSH keys")
    step(f"Ensuring ed25519 key on {args.prefill}")
    p_key = ensure_ssh_key(args.prefill)
    ok(f"Public key: {style(p_key.split()[1][:40] + '…', C.DIM)}")
    step(f"Ensuring ed25519 key on {args.decode}")
    d_key = ensure_ssh_key(args.decode)
    ok(f"Public key: {style(d_key.split()[1][:40] + '…', C.DIM)}")
    step("Cross-installing keys into peers' authorized_keys")
    add_authorized_key(args.decode, p_key, peer_label="prefill")
    add_authorized_key(args.prefill, d_key, peer_label="decode")

    banner("Step 5 / 7 · /etc/hosts entries")
    step(f"On {args.prefill}: add decode pod hostname + 'b200-decode' alias")
    ensure_etc_hosts(args.prefill, d_ipv6, d_host, label="decode pod hostname")
    ensure_etc_hosts(args.prefill, d_ipv6, "b200-decode", label="decode alias")
    step(f"On {args.decode}: add prefill pod hostname + 'b200-prefill' alias")
    ensure_etc_hosts(args.decode, p_ipv6, p_host, label="prefill pod hostname")
    ensure_etc_hosts(args.decode, p_ipv6, "b200-prefill", label="prefill alias")

    banner("Step 6 / 7 · ~/.ssh/config aliases (auto-accept new host keys)")
    ensure_ssh_config_alias(args.prefill, "b200-decode")
    ensure_ssh_config_alias(args.prefill, d_host)
    ensure_ssh_config_alias(args.decode, "b200-prefill")
    ensure_ssh_config_alias(args.decode, p_host)

    banner("Step 7 / 7 · Verify SSH works both ways")
    p2d_alias = test_ssh(args.prefill, "b200-decode", "decode (via alias)")
    p2d_host_ok = test_ssh(args.prefill, d_host, "decode (via pod hostname)")
    d2p_alias = test_ssh(args.decode, "b200-prefill", "prefill (via alias)")
    d2p_host_ok = test_ssh(args.decode, p_host, "prefill (via pod hostname)")
    if not (p2d_alias and p2d_host_ok and d2p_alias and d2p_host_ok):
        fatal("SSH connectivity verification failed; see errors above.")

    step("Quick GPU sanity check")
    gpu_check(args.prefill)
    gpu_check(args.decode)

    banner("🎉 Setup complete!")
    print()
    print(f"  Prefill: {style(args.prefill, C.BOLD)}  ({p_host})")
    print(f"  Decode:  {style(args.decode, C.BOLD)}  ({d_host})")
    print()
    print(
        style(
            "  To launch the DI runtime, paste this into the prefill node:",
            C.BOLD,
        )
    )
    print()
    snippet = textwrap.dedent(
        f"""
            export DECODE_HOST={d_host}
            export PREFILL_HOST={p_host}

            cd ~/code/modular
            NIXL_LOG_LEVEL=DEBUG \\
            MODULAR_PATH="$PWD" \\
            MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=95 \\
            MODULAR_NIXL_TRANSFER_BACKEND=libfabric \\
            FI_EFA_USE_DEVICE_RDMA=1 \\
            ./bazelw run //max/examples/internal/di:run_dist -- \\
              --config-yaml="$PWD/max/examples/internal/di/configs/deepseekv31-nvfp4-1p1d-internode-efa.yaml" \\
              --start-proxy-server --proxy-port 9999 \\
              --hosts "$DECODE_HOST,$PREFILL_HOST"
        """
    ).strip()
    for line in snippet.splitlines():
        print("    " + style(line, C.GREEN))
    print()
    coder_owner = get_coder_owner()
    print(
        style(
            "  To sync your worktree to the prefill node, paste this on your local machine:",
            C.BOLD,
        )
    )
    print()
    rsync_snippet = textwrap.dedent(
        f"""
            ./utils/rsync_coder_hosts.sh --smart --delete main.{args.prefill}.{coder_owner}.coder
        """
    ).strip()
    for line in rsync_snippet.splitlines():
        print("    " + style(line, C.GREEN))
    print()
    print(
        f"  See {style('run_efa_di_benchmark.md', C.BOLD, C.BRIGHT_CYAN)} next to "
        f"this script for the full benchmark walkthrough and error-checking commands."
    )
    print()


if __name__ == "__main__":
    main()
