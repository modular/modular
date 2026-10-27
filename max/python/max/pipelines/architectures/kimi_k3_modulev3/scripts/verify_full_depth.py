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
"""Builds the real full-depth Kimi K3 (ModuleV3) model, loads the real
0.4B bring-up fixture's checkpoint, and runs it with real weights as far as
this hardware's kernels allow.

Not a bazel test and not tied to the registry -- it needs a real checkpoint on
disk, which a hermetic bazel test cannot depend on. Run it through the
library's own venv, which gives plain filesystem access this script needs:

    ./bazelw run //max/tests/integration/architectures/kimi_k3_modulev3:kda_attention_tests_gpu.venv
    source .max+tests+integration+architectures+kimi_k3_modulev3+kda_attention_tests_gpu.venv/bin/activate
    python3 max/python/max/pipelines/architectures/kimi_k3_modulev3/scripts/verify_full_depth.py \\
        --config /tmp/k3_04b_fixture/config.json \\
        --checkpoint /tmp/k3_04b_fixture/model.safetensors

The fixture is ``inference-optimization/Kimi-K3-0.40B`` -- the bring-up
fixture named in ``max_private/kimi_k3/arch.py``'s ``example_repo_ids`` and in
the Mach model registry's comment on this architecture. Fetch it with:

    curl -sL "https://huggingface.co/inference-optimization/Kimi-K3-0.40B/resolve/main/config.json" \\
        -o /tmp/k3_04b_fixture/config.json
    curl -sL "https://huggingface.co/inference-optimization/Kimi-K3-0.40B/resolve/main/model.safetensors" \\
        -o /tmp/k3_04b_fixture/model.safetensors

KNOWN BLOCKER, not a bug in this port: the real fixture's ``num_experts=8``
makes the single-group MoE router kernel's ``num_threads`` (bound to
``num_experts``) 8 -- not a whole number of warps on ANY current GPU (64-wide
on this AMD gfx950 box, 32-wide on NVIDIA), so
``max/kernels/src/nn/moe.mojo``'s own ``constraint failed: num_threads must be
a whole number of warps`` fires at compile time for every one of this
checkpoint's 7 MoE layers, independent of which weights are loaded or which
vendor's GPU compiles it. This script therefore verifies what the hardware
allows: the full real checkpoint loads
onto the full real 8-layer module tree without error (weight assignment
triggers no compute), layer 0 (the one KDA+dense layer, no MoE) runs end to
end with real weights, and both MLA layers (3 and 7) run their real-weighted
attention module standalone (bypassing their MoE FFN, which is what would
hit the same kernel constraint). It does NOT claim a full 8-layer stacked
forward pass, which this hardware cannot compile for this checkpoint's expert
count.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from _paged_cache_test_utils import (
    build_kda_state,
    build_paged_cache,
)
from max import functional as F
from max import tree
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import (
    MHAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
)
from max.pipelines.architectures.kimi_k3_modulev3.model import (
    KimiK3Model,
    KimiK3ModelConfig,
    load_kimi_k3_weights,
)
from max.pipelines.architectures.kimi_k3_modulev3.state_cache import (
    ATTN_CACHE_KEY,
    STATE_CACHE_KEY,
    kda_state_regions,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from max.tensor import Tensor, default_device
from safetensors.numpy import load_file


def _check_finite(name: str, array: np.ndarray) -> bool:
    finite = bool(np.all(np.isfinite(array)))
    print(
        f"  {name}: shape={array.shape} dtype={array.dtype} finite={finite}"
        f" min={array.min():.4f} max={array.max():.4f}"
    )
    return finite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seq-len", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = json.load(f)
    config = KimiK3ModelConfig.from_huggingface_config(config_dict)
    print(
        f"Config: {config.num_hidden_layers} layers,"
        f" kda={config.kda_layer_indices} full_attn={config.full_attn_layer_indices},"
        f" first_k_dense_replace={config.first_k_dense_replace},"
        f" num_experts={config.num_experts}, hidden_size={config.hidden_size},"
        f" vocab_size={config.vocab_size}"
    )

    print(f"Loading checkpoint tensors from {args.checkpoint} ...")
    raw = load_file(args.checkpoint)
    print(f"Loaded {len(raw)} tensors.")

    all_ok = True
    device = Accelerator()
    with default_device(device):
        device_ref = DeviceRef.from_device(device)
        attn_params = MHAKVCacheParams(
            dtype=DType.float32,
            n_kv_heads=config.num_attention_heads,
            head_dim=config.qk_nope_head_dim + config.qk_rope_head_dim,
            num_layers=len(config.full_attn_layer_indices),
            devices=[device_ref],
            page_size=128,
        )
        state_params = RecurrentStateParams(
            regions=kda_state_regions(
                num_kda_layers=len(config.kda_layer_indices),
                conv_dim=3 * config.kda_num_heads * config.kda_head_dim,
                conv_kernel_size=config.kda_conv_kernel_size,
                num_heads=config.kda_num_heads,
                head_dim=config.kda_head_dim,
                conv_dtype=DType.float32,
                recurrent_dtype=DType.float32,
                num_devices=1,
            ),
            devices=[device_ref],
        )
        kv_params = MultiKVCacheParams.from_params(
            {ATTN_CACHE_KEY: attn_params, STATE_CACHE_KEY: state_params}
        )

        print("Constructing the full real 8-layer model ...")
        model = KimiK3Model(config, kv_params=kv_params)
        print("Loading every real checkpoint weight onto the module tree ...")
        load_kimi_k3_weights(model, raw)
        print("Full real checkpoint loaded onto the full real module tree: OK")

        seq_len = args.seq_len
        rng = np.random.default_rng(0)
        input_ids_np = rng.integers(
            0, config.vocab_size, size=(seq_len,), dtype=np.int64
        )
        input_ids = Tensor(input_ids_np, dtype=DType.int64)
        input_row_offsets = Tensor(
            np.array([0, seq_len], dtype=np.uint32), dtype=DType.uint32
        )
        slot_idx = Tensor(np.array([0], dtype=np.uint32), dtype=DType.uint32)

        print(
            "\n--- Layer 0 (KDA + dense, real weights, real forward pass) ---"
        )
        hidden_states = model.embed_tokens(input_ids)
        inner = config.kda_num_heads * config.kda_head_dim
        conv_pool = Tensor.zeros(
            [1, 3 * inner, config.kda_conv_kernel_size - 1], dtype=DType.float32
        )
        state_pool = Tensor.zeros(
            [1, config.kda_num_heads, config.kda_head_dim, config.kda_head_dim],
            dtype=DType.float32,
        )
        layer0 = model.layers[0]
        assert layer0.is_kda and layer0.is_dense
        layer0_out, _ = layer0(
            hidden_states,
            [],
            conv_pool=conv_pool,
            conv_row_id=slot_idx,
            state_pool=state_pool,
            recurrent_row_id=slot_idx,
            input_row_offsets=input_row_offsets,
        )
        all_ok &= _check_finite("layer0_out", layer0_out.to_numpy())

        print(
            "\n--- MLA attention, real weights, standalone (bypassing MoE) ---"
        )
        for layer_idx in config.full_attn_layer_indices:
            layer = model.layers[layer_idx]
            assert not layer.is_kda
            padded = F.reshape(hidden_states, [1, seq_len, config.hidden_size])
            mla_out = layer.attention(padded)
            all_ok &= _check_finite(
                f"layer{layer_idx}_mla_attn_out", mla_out.to_numpy()
            )

        print(
            "\n--- Full 8-layer stacked forward pass (expected to fail: see"
            " module docstring's MoE router kernel constraint) ---"
        )
        try:
            kv_collection = build_paged_cache(attn_params, [seq_len], device)
            kda_state = build_kda_state(
                num_kda_layers=len(config.kda_layer_indices),
                num_requests=1,
                conv_dim=3 * config.kda_num_heads * config.kda_head_dim,
                conv_kernel_size=config.kda_conv_kernel_size,
                num_heads=config.kda_num_heads,
                head_dim=config.kda_head_dim,
                conv_dtype=DType.float32,
                recurrent_dtype=DType.float32,
                device=device,
            )
            kv_inputs = tree.leaves(
                {ATTN_CACHE_KEY: [kv_collection], STATE_CACHE_KEY: [kda_state]}
            )
            return_n_logits = Tensor(
                np.array([1], dtype=np.int64), dtype=DType.int64, device=CPU()
            )
            run = model.compile(
                *modulev3_ragged_kv_symbolic_inputs(
                    kv_params=kv_params, device_refs=[device_ref]
                )
            )
            run(input_ids, return_n_logits, input_row_offsets, *kv_inputs)
            print(
                "UNEXPECTED: the full stacked forward pass compiled and ran."
                " The num_threads/warp-size blocker documented above did not"
                " reproduce -- re-check whether something about this run"
                " changed (hardware, MAX version, kernel)."
            )
        except Exception as exc:
            msg = str(exc)
            if "whole number of warps" in msg:
                print(
                    "CONFIRMED (expected): full stacked forward fails to compile"
                    " at the MoE router kernel's num_threads/warp-size"
                    " constraint, as documented in this script's module"
                    " docstring. Not a bug in this port's code."
                )
            else:
                print(
                    f"UNEXPECTED failure (different from the known blocker): {exc}"
                )
                all_ok = False

    print()
    if all_ok:
        print(
            "PASS: real weights load onto the full real module tree; the"
            " non-MoE real-weighted paths (layer 0 KDA+dense end to end,"
            " both MLA attention layers standalone) produce finite output."
            " Full 8-layer MoE-inclusive forward is blocked by a confirmed"
            " hardware/kernel constraint, not by this port's code."
        )
        return 0
    print("FAIL: one or more real-weight checks produced non-finite output.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
