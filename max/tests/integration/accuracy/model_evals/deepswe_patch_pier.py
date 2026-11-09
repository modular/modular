# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Patch the installed Pier (datacurve-pier 0.3.x) for our hosts. Idempotent.

1. Egress proxy ports. Pier sends the agent's only egress through a squid
   proxy whose Safe_ports ACL admits 80 and 443. A MAX Serve on another port
   (9000 for the self-launched server in CI, 8000 locally) must be admitted
   too; the proxy still limits destinations to the allowlisted model host.

2. Container oom_score_adj. Podman's Docker-compatible API sets OomScoreAdj=0
   on every container it creates. Inside a Kubernetes pod the host process
   already sits above 0 and lacks CAP_SYS_RESOURCE, so crun cannot lower it
   and every `docker compose up` fails with "write to /proc/self/oom_score_adj:
   Permission denied". Carrying the inherited value into Pier's compose specs
   (agent, verifier and proxy containers) makes the write a no-op. A no-op on
   hosts that run at 0.

3. Proxy PID 1. squid switches to its unprivileged user and becomes
   non-dumpable; exec'ing into a container whose PID 1 is non-dumpable needs
   CAP_SYS_PTRACE, which the CI runner pods lack, so the proxy's healthcheck
   never passed there. Run squid as a child of bash instead.

usage: <pier venv python> -I patch_pier.py --ports 8000 9000
"""

import argparse
import re
import sys

import pier.environments.docker as docker_pkg
from pier.environments import agent_setup

HELPER = '''

def _inherited_oom_score_adj() -> int:
    """oom_score_adj of this process, so compose-created containers keep it
    (podman's compat API would otherwise set 0, which an unprivileged pod
    process is not allowed to write). 0 means no override."""
    try:
        with open("/proc/self/oom_score_adj") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0
'''


def patch_ports(path: str, ports: list[str]) -> None:
    src = open(path).read()
    line = "acl Safe_ports port 80 443" + "".join(f" {p}" for p in ports) + "\n"
    new, n = re.subn(r"acl Safe_ports port 80 443[^\n]*\n", line, src, count=1)
    if n != 1:
        sys.exit(f"Safe_ports ACL not found in {path}")
    if new == src:
        print(f"Already patched (ports): {path}")
        return
    open(path, "w").write(new)
    print(f"Patched (ports {' '.join(ports)}): {path}")


def patch_oom(
    path: str, anchor_import: str, anchor_use: str, insert_after_use: str
) -> None:
    src = open(path).read()
    if "_inherited_oom_score_adj" in src:
        print(f"Already patched (oom_score_adj): {path}")
        return
    if src.count(anchor_import) != 1 or src.count(anchor_use) != 1:
        sys.exit(
            f"oom_score_adj patch anchors not found exactly once in {path}"
        )
    src = src.replace(anchor_import, anchor_import + HELPER, 1)
    src = src.replace(anchor_use, anchor_use + insert_after_use, 1)
    open(path, "w").write(src)
    print(f"Patched (oom_score_adj): {path}")


SQUID_EXEC = "exec squid -N -f /tmp/squid.conf -d 1\n"
# bash stays PID 1 and forwards TERM so compose can still stop the proxy.
SQUID_CHILD = (
    "squid -N -f /tmp/squid.conf -d 1 &\n"
    "squid_pid=$!\n"
    "trap 'kill -TERM \"$squid_pid\" 2>/dev/null' TERM INT\n"
    'wait "$squid_pid"\n'
)


def patch_squid_pid1(path: str) -> None:
    """Keep bash, not squid, as PID 1 of the egress proxy container.

    squid switches to its unprivileged user, which marks the process
    non-dumpable, and joining a non-dumpable PID 1's namespaces needs
    CAP_SYS_PTRACE. The CI runner pods lack it, so every ``podman exec`` and
    healthcheck against the proxy died with "container create failed (no logs
    from conmon)" and the agent, which waits on the proxy's health, never
    started. Dev boxes run privileged and never see this.
    """
    src = open(path).read()
    if SQUID_CHILD in src:
        print(f"Already patched (squid pid 1): {path}")
        return
    if src.count(SQUID_EXEC) != 1:
        sys.exit(f"squid exec line not found exactly once in {path}")
    open(path, "w").write(src.replace(SQUID_EXEC, SQUID_CHILD, 1))
    print(f"Patched (squid pid 1): {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ports",
        nargs="+",
        default=["8000", "9000"],
        help="extra ports squid admits",
    )
    args = parser.parse_args()

    patch_ports(agent_setup.__file__, args.ports)
    patch_squid_pid1(agent_setup.__file__)
    patch_oom(
        docker_pkg.__file__,
        "from pier.models.trial.config import ServiceVolumeConfig\n",
        '    main = {"deploy": {"resources": resources}} if resources else {}\n',
        "    if (_oom := _inherited_oom_score_adj()) > 0:\n"
        '        main["oom_score_adj"] = _oom\n',
    )
    patch_oom(
        agent_setup.__file__,
        "def new_proxy_token() -> str:\n    return secrets.token_urlsafe(24)\n",
        '                "networks": ["pier-egress-internal", "default"],\n'
        "            },\n"
        "        },\n"
        '        "networks": {\n'
        '            "pier-egress-internal": {\n'
        '                "internal": True,\n'
        "            },\n"
        "        },\n"
        "    }\n",
        "    if (_oom := _inherited_oom_score_adj()) > 0:\n"
        '        compose["services"][EGRESS_PROXY_SERVICE]["oom_score_adj"] = _oom\n',
    )


if __name__ == "__main__":
    main()
