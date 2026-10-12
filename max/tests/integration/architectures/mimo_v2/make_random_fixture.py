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
"""Random-weight MiMo-V2 fixtures at production width, with HF references.

Writes two checkpoints in the NVFP4 export's format, each with three decoder
layers taken from the real schedule (layer 0: full attention and the dense
MLP; layer 1: SWA and MoE; layer 5: full attention and MoE), all 256 experts
and every width at its production value:

* ``random``: weights drawn to match the real checkpoint's per-tensor
  statistics. Dense F32 weights are exact FP8 128x128 block values and the
  experts are NVFP4 codes with MXFP4-derived scales, as in the export.
* ``beacon``: ``random`` with two beacons that make BF16-invisible errors
  visible. Token ``BEACON`` gets a large one-hot embedding and layer 1 a large
  K weight on that dim, so the beacon's key dominates every query that can see
  it: a query 128 rows after a beacon must not (the window's 128 keys include
  the query), and a 129-key window would. Layer 2's router weight is zero, so
  the correction bias alone picks the top 8; sixteen experts sit 2^-12 apart
  inside one BF16 ulp, which F32 separates and a BF16 bias ties.

Then runs the checkpoint's own ``modeling_mimo_v2.py`` (eager attention, TF32
off) and writes gold-format outputs: ``random`` in F32 (the gold) and BF16
(its self-noise), ``beacon`` in F32 and two F32 mutants, a 129-key window and
a BF16-rounded correction bias, which the beacons must expose.

It needs the published checkpoint's config and modeling code and a GPU, so
it runs by hand, never in CI. Score MAX with ``check_reference_fixtures.py``.

With ``--tiny-goldens``, it instead runs the modeling code in float32 on the
CPU on the ``tiny_fixture`` checkpoint and writes the goldens that
``test_serving_numerics_gpu.py`` scores MAX on: every row's logits for each
prompt, with the fixture's digest. Rerun it whenever ``tiny_fixture.py``
changes:

    make_random_fixture --real <NVFP4 export> \
        --tiny-goldens testdata/tiny_goldens.npz
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tiny_fixture
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

LAYERS = [(0, 0), (1, 1), (0, 1)]
"""(sliding, MoE) per layer: the real layers 0, 1 and 5."""
H, EXPERTS, MOE_DIM, DENSE_DIM, VOCAB = 4096, 256, 2048, 16384, 152576
PROMPT_LENGTHS = (64, 200, 2000)
BEACON, BEACON_DIM, BEACON_ROWS = 1000, 7, (10, 50, 90)
E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
)


def _normal(
    rng: np.random.Generator,
    shape: int | Sequence[int],
    mean: float = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    return (rng.standard_normal(shape, dtype=np.float32) * std + mean).astype(
        np.float32
    )


def _bf16(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, np.float32)).bfloat16()


def fp8_exact(w: np.ndarray) -> np.ndarray:
    """F32 values that 128x128-block FP8 E4M3 represents exactly."""
    r, c = w.shape
    tiles = torch.from_numpy(w).reshape(r // 128, 128, c // 128, 128)
    scale = tiles.abs().amax(dim=(1, 3), keepdim=True) / 448.0
    codes = (tiles / scale).to(torch.float8_e4m3fn)
    return (codes.float() * scale).reshape(r, c).numpy()


def _qkv_rows(sliding: bool) -> tuple[int, int, int]:
    return (3072, 384, 256) if sliding else (3072, 192, 128)


def _to_global(chunks: np.ndarray, sliding: bool) -> np.ndarray:
    """Chunk order ``[4, q + k + v, H]`` -> the export's global ``[Q; K; V]``."""
    q, k, _ = _qkv_rows(sliding)
    return np.concatenate(
        [
            chunks[:, :q].reshape(-1, H),
            chunks[:, q : q + k].reshape(-1, H),
            chunks[:, q + k :].reshape(-1, H),
        ]
    )


def _to_chunks(weight: np.ndarray, sliding: bool) -> np.ndarray:
    q, k, v = _qkv_rows(sliding)
    return np.concatenate(
        [
            weight[: 4 * q].reshape(4, q, H),
            weight[4 * q : 4 * (q + k)].reshape(4, k, H),
            weight[4 * (q + k) :].reshape(4, v, H),
        ],
        axis=1,
    )


def _exact_qkv(chunks: np.ndarray, sliding: bool) -> np.ndarray:
    """Makes chunk-order rows FP8-exact per block, pad rows included."""
    rows = chunks.shape[1]
    padded = -(-rows // 128) * 128
    full = np.zeros((4, padded, H), np.float32)
    full[:, :rows] = chunks
    full = fp8_exact(full.reshape(-1, H)).reshape(4, padded, H)
    return _to_global(full[:, :rows], sliding)


def _export_scales(e8m0: np.ndarray) -> np.ndarray:
    """The converter's scales: E8M0 per 32 -> two E4M3 per 16, global 2^-8."""
    exp = e8m0.astype(np.int16) - 127 + 8
    assert exp.min() >= -9 and exp.max() <= 8
    out = np.where(exp >= -6, (exp + 7) << 3, 1 << np.clip(exp + 9, 0, 7))
    return np.repeat(out.astype(np.uint8), 2, axis=-1)


def make_random(real: Path, out: Path) -> None:
    rng = np.random.default_rng(20260926)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((real / "config.json").read_text())
    config.update(
        num_hidden_layers=len(LAYERS),
        hybrid_layer_pattern=[s for s, _ in LAYERS],
        moe_layer_freq=[m for _, m in LAYERS],
        vision_config=None,
        audio_config=None,
    )
    config["quantization_config"]["quantized_layers"] = {
        f"model.layers.{i}.mlp.experts.{e}.{p}": {
            "quant_algo": "W4A16_NVFP4",
            "group_size": 16,
        }
        for i, (_, moe) in enumerate(LAYERS)
        if moe
        for e in range(EXPERTS)
        for p in ("gate_proj", "up_proj", "down_proj")
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))
    for f in ("configuration_mimo_v2.py", "modeling_mimo_v2.py"):
        shutil.copy(real / f, out / f)

    save_file(
        {
            "model.embed_tokens.weight": _bf16(
                _normal(rng, (VOCAB, H), std=0.0165)
            ),
            "lm_head.weight": _bf16(_normal(rng, (VOCAB, H), std=0.011)),
            "model.norm.weight": _bf16(_normal(rng, H, 3.6, 0.34)),
        },
        str(out / "model-embed.safetensors"),
    )
    for i, (sliding, moe) in enumerate(LAYERS):
        p = f"model.layers.{i}."
        q, k, v = _qkv_rows(bool(sliding))
        t = {
            p + "input_layernorm.weight": _bf16(_normal(rng, H, 0.3, 0.1)),
            p + "post_attention_layernorm.weight": _bf16(
                _normal(rng, H, 0.3, 0.1)
            ),
            p + "self_attn.qkv_proj.weight": torch.from_numpy(
                _exact_qkv(
                    _normal(rng, (4, q + k + v, H), std=0.0078), bool(sliding)
                )
            ),
            p + "self_attn.o_proj.weight": _bf16(
                _normal(rng, (H, 64 * 128), std=0.009)
            ),
        }
        if sliding:
            t[p + "self_attn.attention_sink_bias"] = _bf16(
                _normal(rng, 64, 0.8, 0.5)
            )
        if not moe:
            for proj, shape in (
                ("gate_proj", (DENSE_DIM, H)),
                ("up_proj", (DENSE_DIM, H)),
                ("down_proj", (H, DENSE_DIM)),
            ):
                t[p + f"mlp.{proj}.weight"] = torch.from_numpy(
                    fp8_exact(_normal(rng, shape, std=0.0096))
                )
        else:
            t[p + "mlp.gate.weight"] = _bf16(
                _normal(rng, (EXPERTS, H), std=0.028)
            )
            t[p + "mlp.gate.e_score_correction_bias"] = torch.from_numpy(
                _normal(rng, EXPERTS, 2.05, 0.04)
            )
            for e in range(EXPERTS):
                for proj, (n, kk) in (
                    ("gate_proj", (MOE_DIM, H)),
                    ("up_proj", (MOE_DIM, H)),
                    ("down_proj", (H, MOE_DIM)),
                ):
                    b = p + f"mlp.experts.{e}.{proj}."
                    t[b + "weight"] = torch.from_numpy(
                        rng.integers(0, 256, (n, kk // 2), dtype=np.uint8)
                    )
                    e8m0 = rng.integers(116, 122, (n, kk // 32), dtype=np.uint8)
                    t[b + "weight_scale"] = torch.from_numpy(
                        _export_scales(e8m0)
                    ).view(torch.float8_e4m3fn)
                    t[b + "weight_scale_2"] = torch.tensor([2.0**-8])
        save_file(t, str(out / f"model-layer{i}.safetensors"))


def make_beacon(random: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in (
        "config.json",
        "configuration_mimo_v2.py",
        "modeling_mimo_v2.py",
        "model-layer0.safetensors",
    ):
        shutil.copy(random / f, out / f)

    t = _load(random / "model-embed.safetensors")
    emb = t["model.embed_tokens.weight"]
    emb[BEACON] = 0
    emb[BEACON, BEACON_DIM] = 30.0
    save_file(t, str(out / "model-embed.safetensors"))

    t = _load(random / "model-layer1.safetensors")
    t["model.layers.1.input_layernorm.weight"][BEACON_DIM] = 1.0
    name = "model.layers.1.self_attn.qkv_proj.weight"
    q, k, _ = _qkv_rows(True)
    chunks = _to_chunks(t[name].numpy().copy(), True)
    chunks[:, q : q + k, BEACON_DIM] = 1.0
    t[name] = torch.from_numpy(_exact_qkv(chunks, True))
    save_file(t, str(out / "model-layer1.safetensors"))

    t = _load(random / "model-layer2.safetensors")
    t["model.layers.2.mlp.gate.weight"].zero_()
    bias = torch.full((EXPERTS,), 1.0)
    # Experts 16-31: the odd ones win in F32 and the even ones lose, so no
    # index-order tie-break can recover the F32 choice from a BF16 tie.
    for j in range(8):
        bias[16 + 2 * j] = 2.0 + j * 2.0**-12
        bias[17 + 2 * j] = 2.0 + (8 + j) * 2.0**-12
    assert (bias.bfloat16().float()[16:32] == 2.0).all()
    t["model.layers.2.mlp.gate.e_score_correction_bias"] = bias
    save_file(t, str(out / "model-layer2.safetensors"))


def _load(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, "pt") as f:
        # `safe_open` only exposes `.keys()`, not `__iter__`; SIM118's fix
        # doesn't apply here.
        return {n: f.get_tensor(n) for n in f.keys()}  # noqa: SIM118


class _NVFP4Expert(torch.nn.Module):
    """``MiMoV2MLP.forward`` with weights decoded from NVFP4 on each use."""

    def __init__(
        self,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        dtype: torch.dtype,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.act_fn, self.dtype_ = act_fn, dtype
        for name, value in tensors.items():
            self.register_buffer(name, value, persistent=False)

    def weight(self, proj: str) -> torch.Tensor:
        codes = getattr(self, f"{proj}_codes")
        lut = E2M1.to(codes.device)
        # Two E2M1 codes per byte, low nibble first.
        values = torch.stack(
            [lut[(codes & 15).long()], lut[(codes >> 4).long()]], -1
        ).reshape(codes.shape[0], -1)
        scale = getattr(self, f"{proj}_scale").view(torch.float8_e4m3fn)
        scale = scale.float() * getattr(self, f"{proj}_scale_2")
        return (values * scale.repeat_interleave(16, 1)).to(self.dtype_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(F.linear(x, self.weight("gate")))
        return F.linear(
            gate * F.linear(x, self.weight("up")), self.weight("down")
        )


def reference_model(
    checkpoint: Path, dtype: torch.dtype, device: torch.device
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """Builds the checkpoint's own HF model on ``device`` in ``dtype``.

    The routed experts decode their NVFP4 weights on each use. Returns the
    model and a dict that its forward fills with the embedding, each decoder
    layer's output and the final norm's, for the first sequence.
    """
    config = AutoConfig.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        vision_config=None,
        audio_config=None,
    )
    config._attn_implementation = "eager"
    cls = get_class_from_dynamic_module(
        "modeling_mimo_v2.MiMoV2ForCausalLM", str(checkpoint)
    )
    with torch.device("meta"):
        model = cls(config)
    model.eval()
    tensors: dict[str, torch.Tensor] = {}
    for f in sorted(checkpoint.glob("*.safetensors")):
        tensors |= _load(f)

    def put(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
        module._parameters[name] = torch.nn.Parameter(
            value.to(device), requires_grad=False
        )

    m = model.model
    put(
        m.embed_tokens, "weight", tensors["model.embed_tokens.weight"].to(dtype)
    )
    put(m.norm, "weight", tensors["model.norm.weight"].to(dtype))
    put(model.lm_head, "weight", tensors["lm_head.weight"].to(dtype))
    for i, layer in enumerate(m.layers):
        p = f"model.layers.{i}."
        attn = layer.self_attn
        for name in ("qkv_proj", "o_proj"):
            put(
                getattr(attn, name),
                "weight",
                tensors[p + f"self_attn.{name}.weight"].to(dtype),
            )
        if attn.attention_sink_bias is not None:
            put(
                attn,
                "attention_sink_bias",
                tensors[p + "self_attn.attention_sink_bias"].to(dtype),
            )
        for norm in ("input_layernorm", "post_attention_layernorm"):
            put(
                getattr(layer, norm),
                "weight",
                tensors[p + f"{norm}.weight"].to(dtype),
            )
        mlp = layer.mlp
        if not hasattr(mlp, "experts"):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                put(
                    getattr(mlp, proj),
                    "weight",
                    tensors[p + f"mlp.{proj}.weight"].to(dtype),
                )
            continue
        put(mlp.gate, "weight", tensors[p + "mlp.gate.weight"].to(dtype))
        put(
            mlp.gate,
            "e_score_correction_bias",
            tensors[p + "mlp.gate.e_score_correction_bias"],
        )
        experts = []
        for e in range(config.n_routed_experts):
            b = f"{p}mlp.experts.{e}."
            experts.append(
                _NVFP4Expert(
                    mlp.experts[e].act_fn,
                    dtype,
                    {
                        f"{proj}_{field}": tensors[b + f"{proj}_proj.{key}"].to(
                            device
                        )
                        for proj in ("gate", "up", "down")
                        for field, key in (
                            ("codes", "weight"),
                            ("scale", "weight_scale"),
                            ("scale_2", "weight_scale_2"),
                        )
                    },
                )
            )
        mlp.experts = torch.nn.ModuleList(experts)
    rotary = type(m.rotary_emb)
    m.rotary_emb = rotary(config=config, is_swa=False, device=device)
    m.swa_rotary_emb = rotary(config=config, is_swa=True, device=device)
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        assert t.device.type != "meta", name

    captured: dict[str, torch.Tensor] = {}

    def capture(
        name: str,
    ) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], None]:
        def hook(
            module: torch.nn.Module, args: tuple[Any, ...], out: Any
        ) -> None:
            h = out[0] if isinstance(out, tuple) else out
            captured[name] = h[0].detach().float().cpu()

        return hook

    m.embed_tokens.register_forward_hook(capture("embed"))
    for i, layer in enumerate(m.layers):
        layer.register_forward_hook(capture(f"layer_{i:02d}"))
    m.norm.register_forward_hook(capture("final_norm"))
    return model, captured


@torch.no_grad()
def run_reference(
    checkpoint: Path,
    out: Path,
    dtype: torch.dtype,
    prompts: dict[str, list[int]],
    mutate: Callable[[torch.nn.Module], None] | None = None,
) -> None:
    device = torch.device("cuda:0")
    model, captured = reference_model(checkpoint, dtype, device)
    if mutate is not None:
        mutate(model)
    for name, ids in prompts.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        captured.clear()
        hidden = model.model(
            input_ids=torch.tensor(ids, device=device)[None], use_cache=False
        ).last_hidden_state
        logits = model.lm_head(hidden)[0].float().cpu()
        lse = logits.logsumexp(-1)
        values, indices = logits.topk(64, -1)
        target = torch.full((len(ids),), float("nan"))
        target[:-1] = (
            logits[:-1].gather(1, torch.tensor(ids[1:])[:, None])[:, 0]
            - lse[:-1]
        )
        positions = torch.arange(len(ids), dtype=torch.int32)
        save_file(
            {
                "values": values.contiguous(),
                "indices": indices.int().contiguous(),
                "logsumexp": lse,
                "target_logprob": target,
            },
            str(d / "logits_topk.safetensors"),
        )
        save_file(
            {"positions": positions, "logits": logits.contiguous()},
            str(d / "logits_full.safetensors"),
        )
        save_file(
            {"positions": positions}
            | {k: v.contiguous() for k, v in captured.items()},
            str(d / "hidden.safetensors"),
        )
        (d / "seq.json").write_text(
            json.dumps({"ids": ids, "prompt_len": len(ids)})
        )
    del model
    torch.cuda.empty_cache()


@torch.no_grad()
def make_tiny_goldens(real: Path, out: Path) -> None:
    """Writes the HF logits of every ``tiny_fixture`` prompt to ``out``."""
    checkpoint = tiny_fixture.tensors()
    goldens = {"digest": np.array(tiny_fixture.digest(checkpoint))}
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        tiny_fixture.write(directory, checkpoint)
        config = json.loads((directory / "config.json").read_text())
        config["auto_map"] = {
            "AutoConfig": "configuration_mimo_v2.MiMoV2Config",
            "AutoModelForCausalLM": "modeling_mimo_v2.MiMoV2ForCausalLM",
        }
        (directory / "config.json").write_text(json.dumps(config))
        for f in ("configuration_mimo_v2.py", "modeling_mimo_v2.py"):
            shutil.copy(real / f, directory / f)
        model, _ = reference_model(
            directory, torch.float32, torch.device("cpu")
        )
        for name, ids in tiny_fixture.prompts().items():
            hidden = model.model(
                input_ids=torch.tensor(ids)[None], use_cache=False
            ).last_hidden_state
            goldens[f"logits_{name}"] = model.lm_head(hidden)[0].numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, allow_pickle=False, **goldens)


def _round_bias_to_bf16(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "e_score_correction_bias"):
            bias = module.e_score_correction_bias
            bias.data = bias.data.bfloat16().float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        type=Path,
        required=True,
        help="The published checkpoint, for its config and modeling code.",
    )
    parser.add_argument("--out", type=Path, help="Where to write the fixtures.")
    parser.add_argument(
        "--tiny-goldens",
        type=Path,
        help="Write only the goldens of test_serving_numerics_gpu.py here.",
    )
    args = parser.parse_args()
    if args.tiny_goldens is not None:
        make_tiny_goldens(args.real, args.tiny_goldens)
        return
    if args.out is None:
        parser.error("--out is required without --tiny-goldens.")
    random, beacon = args.out / "random", args.out / "beacon"
    if not (random / "checkpoint" / "model-layer2.safetensors").exists():
        make_random(args.real, random / "checkpoint")
    make_beacon(random / "checkpoint", beacon / "checkpoint")

    rng = np.random.default_rng(1)
    prompts = {
        f"rand_{n}": rng.integers(0, 151643, n).tolist() for n in PROMPT_LENGTHS
    }
    run_reference(
        random / "checkpoint", random / "hf_f32", torch.float32, prompts
    )
    run_reference(
        random / "checkpoint", random / "hf_bf16", torch.bfloat16, prompts
    )

    ids = np.random.default_rng(2).integers(0, 151643, 256)
    ids[ids == BEACON] = BEACON + 1
    ids[list(BEACON_ROWS)] = BEACON
    beacon_prompts = {"beacon_256": ids.tolist()}
    for variant, mutate in (
        ("hf_f32", None),
        (
            "hf_f32_window_129",
            lambda m: setattr(m.model.config, "sliding_window", 129),
        ),
        ("hf_f32_bias_bf16", _round_bias_to_bf16),
    ):
        run_reference(
            beacon / "checkpoint",
            beacon / variant,
            torch.float32,
            beacon_prompts,
            mutate,
        )
    (beacon / "beacons.json").write_text(
        json.dumps(
            {
                "prompt": "beacon_256",
                "length": len(ids),
                "token": BEACON,
                "rows": list(BEACON_ROWS),
                "window_edge_rows": [r + 128 for r in BEACON_ROWS],
                "window_layer": 1,
                "router_layer": 2,
            }
        )
    )


if __name__ == "__main__":
    main()
