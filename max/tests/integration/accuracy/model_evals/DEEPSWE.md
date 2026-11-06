# DeepSWE eval

DeepSWE scores long-horizon coding with
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) driven by
[Pier](https://pypi.org/project/datacurve-pier/), the harness the public
leaderboard uses, so the number is comparable to theirs. The `run_deepswe`
toggle of `minimaxM3AgentClassDatasetEval.yaml` runs the same target a dev box
runs:

```bash
# Against a MAX Serve on this host, port 8000; smoke-test on 3 sampled tasks
# first. The first run also installs the tasks, Pier, docker compose and the
# Pier patches under DEEPSWE_HOME (idempotent, see deepswe_setup.sh).
MODEL=MiniMaxAI/MiniMax-M3.1-Flash PORT=8000 DEEPSWE_HOME=~/deepswe SAMPLE_SIZE=3 \
  ./bazelw run //max/tests/integration/accuracy/model_evals:deepswe_eval_local

# Full run (113 tasks); DEEPSWE_CONCURRENCY tasks at once, DEEPSWE_ATTEMPTS per task
MODEL=MiniMaxAI/MiniMax-M3.1-Flash PORT=8000 DEEPSWE_HOME=~/deepswe DEEPSWE_CONCURRENCY=8 \
  ./bazelw run //max/tests/integration/accuracy/model_evals:deepswe_eval_local

# Against a remote server instead
MODEL=... MAX_SERVE_URL=https://my-host OPENAI_API_KEY=... DEEPSWE_HOME=~/deepswe \
  ./bazelw run //max/tests/integration/accuracy/model_evals:deepswe_eval_local
```

The same run on CI, as a client of an already-running deployment (48 tasks at
once; add `-f sample_size=3` for a smoke test):

```bash
# <deployment-url>: base URL of the deployment, without /v1
# <served-model>: the model name the deployment serves
# <client-runner>: a runner with podman and a route to the deployment
gh workflow run minimaxM3AgentClassDatasetEval.yaml --ref <branch> \
  -f max_serve_url=<deployment-url> \
  -f model=<served-model> -f client_runner=<client-runner> \
  -f run_deepswe=true -f run_swe_verified=false -f run_gaia=false -f run_hle=false \
  -f deepswe_attempts=1 -f deepswe_concurrency=48
```

`run_deepswe_local.sh` documents every environment variable. Results land in
`$OUT/deepswe/<trial>/` (agent trajectory with per-turn token counts, verifier
output, `result.json`) and `$OUT/score.json`; `pier view $OUT` browses them. The
agent can print its environment, so the endpoint's API key is scrubbed from
everything under `$OUT` when the run exits (`deepswe_redact.py`). A file that
cannot be scrubbed even as root is deleted and listed in
`$OUT/REMOVED_UNSCRUBBABLE_FILES.txt`, and the run fails only if such a file
cannot be deleted either; a self-launched server needs no real key at all.
Scoring (`deepswe_score.py`) follows the leaderboard: graded trials count,
including agent timeouts, and trials with no verifier reward are excluded as
infra errors. A run with no graded trial fails.

Host requirements, all handled or diagnosed by the scripts: a podman or docker
engine (`DOCKER_HOST`), `uv`, `curl`, `git`, `sudo` for the podman socket, and
a writable `/proc/sys` (Pier's sealed agent network needs one sysctl write; on
a non-privileged Kubernetes pod, `sudo mount -o remount,bind,rw /proc/sys` once
per pod lifetime). A server on this host is reached as
`host.containers.internal`, which only Podman defines; with Docker, set
`MAX_SERVE_URL` to an address the containers can reach, such as the host's IP.
Each task holds a 2-CPU, 8 GB agent container and then a
verifier of the same size, for up to 3 hours.

`deepswe_patch_pier.py` carries the three Pier patches the scripts apply:
admitting the model port in the egress proxy, keeping containers startable on
hosts whose `oom_score_adj` cannot be lowered, and running squid as a child of
bash so the proxy container can be exec'd into on hosts without
`CAP_SYS_PTRACE`.
