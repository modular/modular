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
"""MAX's dynamic MXFP8 activation quantize against vLLM's, on one GPU.

Each arm is a device graph of `--copies` quantizes, each on its own input so
no call reads another's from L2, replayed in host-timed bursts; arms
alternate order each round. MAX runs `quantize_dynamic_block_scaled` (BF16 to
E4M3 with E8M0 scales in the tcgen05 layout); vLLM runs `mxfp8_e4m3_quantize`
with swizzled scales, as its MXFP8 linears do, which on SM100 is FlashInfer's
CuTe DSL `mxfp8_quantize`. Prints one `CSV,` line per shape and arm.
"""

import argparse
import importlib
import os
import statistics
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

_TMP = os.environ.get("TEST_TMPDIR", "/tmp")
for _k, _sub in (
    ("FLASHINFER_WORKSPACE_BASE", "fi"),
    ("TRITON_CACHE_DIR", "triton"),
    ("CUTE_DSL_CACHE_DIR", "cute"),
):
    os.environ.setdefault(_k, os.path.join(_TMP, _sub))
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
if os.path.isdir("/usr/local/cuda"):
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ["PATH"] = "/usr/local/cuda/bin" + os.pathsep + os.environ["PATH"]

import torch


def _merge_cutlass_dsl() -> None:
    """Links the nvidia-cutlass-dsl wheels into one package, as pip does.

    Its libs wheels share one `nvidia_cutlass_dsl` package and a `.pth`
    puts its `dsl_packages` on the path, but Bazel keeps each wheel in its
    own tree and skips `.pth` files. Of the CUDA builds, only torch's
    major version is linked.
    """
    other = "cu12" if torch.version.cuda.startswith("13") else "cu13"
    srcs = [
        os.path.join(p, "nvidia_cutlass_dsl")
        for p in sys.path
        if os.path.isfile(os.path.join(p, "nvidia_cutlass_dsl", "__init__.py"))
        and f"libs-{other}@" not in p
    ]
    if not srcs:
        return
    root = os.path.join(_TMP, "cutlass_dsl")
    dst = os.path.join(root, "nvidia_cutlass_dsl")
    for src in srcs:
        for d, _, files in os.walk(src):
            out = os.path.join(dst, os.path.relpath(d, src))
            os.makedirs(out, exist_ok=True)
            for f in files:
                link = os.path.join(out, f)
                if not os.path.lexists(link):
                    os.symlink(os.path.join(d, f), link)
    sys.path[:0] = [os.path.join(dst, "dsl_packages"), root]


_merge_cutlass_dsl()

import gpu_telemetry
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import quantize_dynamic_block_scaled

# vLLM is optional: the `//:use_vllm_setting` select can leave it out.
_mxfp8_utils: ModuleType | None
try:
    _mxfp8_utils = importlib.import_module(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils"
    )
except Exception:
    traceback.print_exc()
    _mxfp8_utils = None


def ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v]


@dataclass
class Replay:
    """A captured graph's replay and the buffers it reads and writes."""

    run: Callable[[], object]
    keep: tuple[object, ...]

    def __call__(self) -> None:
        self.run()


def max_arm(session: InferenceSession, m: int, k: int, copies: int) -> Replay:
    dev = DeviceRef.GPU()
    types = [TensorType(DType.bfloat16, (m, k), device=dev)] * copies
    with Graph(f"mxfp8_quant_{m}_{k}_{copies}", input_types=types) as g:
        outs = []
        for v in g.inputs:
            q, s = quantize_dynamic_block_scaled(
                v.tensor,
                sf_vector_size=32,
                scales_type=DType.float8_e8m0fnu,
                out_type=DType.float8_e4m3fn,
            )
            outs += [q, s]
        g.output(*outs)
    model = session.load(g)
    xs = [
        torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        for _ in range(copies)
    ]
    inputs = [Buffer.from_dlpack(x) for x in xs]
    key = abs(hash((m, k, copies))) % (1 << 31)
    model.capture(key, *inputs)
    for _ in range(3):
        model.replay(key, *inputs)
    torch.cuda.synchronize()

    return Replay(lambda: model.replay(key, *inputs), (model, inputs, xs))


def torch_arm(
    fn: Callable[[torch.Tensor], object], m: int, k: int, copies: int
) -> Replay:
    xs = [
        torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        for _ in range(copies)
    ]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for x in xs:
            fn(x)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [fn(x) for x in xs]
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    return Replay(graph.replay, (graph, xs, outs))


def vllm_fns() -> dict[str, Callable[[torch.Tensor], object]]:
    fns: dict[str, Callable[[torch.Tensor], object]] = {}
    utils = _mxfp8_utils
    if utils is not None:
        fns["vllm"] = lambda x: utils.mxfp8_e4m3_quantize(
            x, is_sf_swizzled_layout=True
        )
    return fns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", default="96,126,1024,4096,6941,8192")
    parser.add_argument("--ks", default="1280,2304,4096,5120,8192")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--copies_bytes", type=int, default=1 << 30)
    args = parser.parse_args()

    session = InferenceSession(devices=[Accelerator()])
    fns = vllm_fns()
    print("CSV,m,k,arm,us,p10,p90,copies", flush=True)
    for k in ints(args.ks):
        for m in ints(args.ms):
            copies = max(1, min(16, args.copies_bytes // (m * k * 2)))
            arms = {"max": max_arm(session, m, k, copies)}
            for name, fn in fns.items():
                try:
                    arms[name] = torch_arm(fn, m, k, copies)
                except Exception:
                    traceback.print_exc()
            times: dict[str, list[float]] = {a: [] for a in arms}
            names = list(arms)
            for rnd in range(args.rounds):
                for name in names if rnd % 2 == 0 else names[::-1]:
                    times[name] += gpu_telemetry.time_bursts(
                        arms[name], copies, args.replays, args.repeats
                    )
            for name, ts in times.items():
                ts = sorted(ts)
                print(
                    f"CSV,{m},{k},{name},{statistics.median(ts):.3f},"
                    f"{ts[len(ts) // 10]:.3f},{ts[(len(ts) * 9) // 10]:.3f},"
                    f"{copies}",
                    flush=True,
                )
            del arms
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
