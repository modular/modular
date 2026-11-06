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
"""Generates a random-weight K3 DSpark drafter checkpoint.

Writes a ``config.json`` + ``model.safetensors`` pair in the published
``k3_dspark`` format, sized for
end-to-end spec-decode tests against the 4-layer Kimi-K3 fixture. Hidden size
and vocabulary must match the target — the drafter embeds its block through
its own embedding and scores every slot through the target's LM head — but
the tap count is just ``len(target_layer_ids)``
(the reference implementation builds ``context_proj`` as
``[num_taps * target_hidden, hidden]``),
so any tap count is expressible. The default remaps the 93-layer, 5-tap
drafter onto the 4-layer fixture as 3 taps ``[0, 1, 2]``: no default tap
names the last layer, whose mixture feature no consumer layer scores.

A dev tool the golden replay test also imports (its weight draw
regenerates the golden's recorded weights from the recorded seed). It
borrows ``torch_reference`` from the dspark test package, so run it
through its bazel target, which carries that dependency::

    ./bazelw run //max/tests/integration/architectures/kimi_k3_modulev3/tools:make_tiny_drafter -- \
        --out /tmp/tiny-dspark --num-layers 2 \
        --target-config <a kimi-k3 config.json>

``--target-config`` is optional when ``--target-hidden-size`` and
``--vocab-size`` are given instead.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections.abc import Mapping, Sequence

import torch
from safetensors.torch import save_file
from torch_reference import expected_state_dict_shapes

# The published checkpoint's special-token pair, kept verbatim where they
# fit the vocabulary.
_PUBLISHED_SPECIAL_TOKENS = {"bos_token_id": 163584, "eos_token_id": 163586}


def read_target_dims(path: pathlib.Path) -> tuple[int, int, int]:
    """Reads the target dims from a ``config.json``.

    Returns ``(hidden_size, vocab_size, num_hidden_layers)`` — Kimi-K3
    nests them under ``text_config``.
    """
    with open(path) as f:
        raw = json.load(f)
    assert isinstance(raw, dict), f"{path}: expected a JSON object"
    for cfg in (raw, raw.get("text_config", {})):
        if isinstance(cfg, dict) and {
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
        } <= set(cfg):
            return (
                int(cfg["hidden_size"]),
                int(cfg["vocab_size"]),
                int(cfg["num_hidden_layers"]),
            )
    raise ValueError(
        f"{path}: no hidden_size/vocab_size/num_hidden_layers at the top "
        f"level or under text_config"
    )


def build_tiny_config(
    *,
    target_hidden_size: int,
    vocab_size: int,
    target_layer_ids: Sequence[int],
    target_num_hidden_layers: int | None = None,
    num_layers: int = 2,
    num_attention_heads: int = 64,
    intermediate_size: int = 1024,
    q_lora_rank: int = 1536,
    kv_lora_rank: int = 512,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    markov_rank: int = 64,
    mask_token_id: int | None = None,
) -> dict[str, object]:
    """Builds the tiny drafter's ``config.json`` content.

    The draft runs at the target's width (it embeds through its own
    embedding, ``has_own_embed_tokens`` in the drafter, and scores every
    slot through the target's LM head); everything else shrinks. Head
    geometry defaults mirror the published drafter so the shared 576-wide
    MLA latent is preserved.

    Args:
        target_hidden_size: The target model's hidden size.
        vocab_size: The target model's vocabulary.
        target_layer_ids: 0-based post-layer taps (MAX's convention), any
            count.
        target_num_hidden_layers: The target's depth; the loader reads it
            back as ``num_target_layers``, which the pairing gate compares
            against the target.
        num_layers: Draft decoder layers.
        num_attention_heads: Draft MLA heads.
        intermediate_size: Draft SwiGLU width.
        q_lora_rank: Draft Q-LoRA rank.
        kv_lora_rank: Draft KV-LoRA rank (the shared latent's width).
        qk_nope_head_dim: Per-head nope width.
        qk_rope_head_dim: Per-head rope width.
        v_head_dim: Per-head value width.
        markov_rank: VanillaMarkov rank; the config parser refuses 0, so
            the head is always built.
        mask_token_id: The mask token; defaults to ``vocab_size - 3``, the
            published drafter's slot.

    Returns:
        The ``config.json`` mapping, with the published checkpoint's key set.
    """
    layer_ids = [int(i) for i in target_layer_ids]
    assert layer_ids and all(i >= 0 for i in layer_ids), (
        f"target_layer_ids must be non-empty and non-negative, got {layer_ids}"
    )
    assert vocab_size > 3
    mask = vocab_size - 3 if mask_token_id is None else int(mask_token_id)
    assert 0 < mask < vocab_size, (
        f"mask_token_id {mask} outside (0, {vocab_size})"
    )
    return {
        "architectures": ["K3DSparkModel"],
        "model_type": "k3_dspark",
        "hidden_size": target_hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_attention_heads,
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "mla_use_nope": False,
        "mla_use_output_gate": False,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-05,
        "max_position_embeddings": 1048576,
        "rope_theta": 50000.0,
        "num_target_layers": len(layer_ids),
        "target_hidden_size": target_hidden_size,
        # The target's depth, which `_check_pairing` compares against the
        # target; the tap count is `num_target_layers`.
        "target_num_hidden_layers": (
            target_num_hidden_layers
            if target_num_hidden_layers is not None
            else max(layer_ids) + 1
        ),
        "target_layer_ids": layer_ids,
        "mask_token_id": mask,
        "bos_token_id": min(
            _PUBLISHED_SPECIAL_TOKENS["bos_token_id"], vocab_size - 1
        ),
        "eos_token_id": min(
            _PUBLISHED_SPECIAL_TOKENS["eos_token_id"], vocab_size - 1
        ),
        "pad_token_id": vocab_size - 1,
        "markov_rank": markov_rank,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "tie_word_embeddings": False,
        "draft_vocab_size": vocab_size,
        "_torchspec_version": "0.1.0",
        "torch_dtype": "bfloat16",
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 50000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    }


_NORM_WEIGHTS = ("context_norm.weight", "final_norm.weight")
_EMBEDDING_WEIGHTS = ("embed_tokens.weight", "markov_head.markov_w1.weight")


def _random_tensor(
    name: str, shape: tuple[int, ...], generator: torch.Generator
) -> torch.Tensor:
    """One random fp32 tensor, drawn in torch's own init distributions.

    RMSNorm weights start at ones; embeddings at N(0, 1); every linear and
    bias at U(-1/sqrt(fan_in), +1/sqrt(fan_in)) (``nn.Linear``'s default) so
    activations stay bounded through the stack.
    """
    if name.endswith("layernorm.weight") or name in _NORM_WEIGHTS:
        return torch.ones(shape, dtype=torch.float32)
    if name in _EMBEDDING_WEIGHTS:
        return torch.empty(shape, dtype=torch.float32).normal_(
            0.0, 1.0, generator=generator
        )
    bound = 1.0 / math.sqrt(shape[-1])
    return torch.empty(shape, dtype=torch.float32).uniform_(
        -bound, bound, generator=generator
    )


def draw_state_dict(
    config: Mapping[str, object],
    *,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Draws the checkpoint's random bf16 weights from ``seed``.

    Tensor names and shapes come from :func:`expected_state_dict_shapes`, the
    same mapping the reference validates against, so the generator cannot
    drift from the loader. Draws happen in sorted-name order under one
    seeded generator, so a seed reproduces the checkpoint bit for bit --
    the golden replay test regenerates the exact weights the reference
    consumed from here.
    """
    shapes = expected_state_dict_shapes(config)
    generator = torch.Generator().manual_seed(seed)
    return {
        name: _random_tensor(name, shapes[name], generator).to(torch.bfloat16)
        for name in sorted(shapes)
    }


def write_checkpoint(
    config: Mapping[str, object],
    out_dir: pathlib.Path | str,
    *,
    seed: int = 0,
) -> pathlib.Path:
    """Writes ``config.json`` and random bf16 ``model.safetensors``."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = draw_state_dict(config, seed=seed)
    save_file(state, str(out / "model.safetensors"))
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    return out


def _default_taps(num_target_layers: int) -> list[int]:
    """Up to five taps spread over every layer but the target's last.

    ``KimiK3Model``'s mixture tap reads the consumer layer after the
    tapped one, and the last layer has none, so no default tap names it.
    """
    tappable = num_target_layers - 1
    if tappable <= 5:
        return list(range(tappable))
    return [round(i * (tappable - 1) / 4) for i in range(5)]


def main(argv: Sequence[str] | None = None) -> None:
    """Writes the tiny drafter checkpoint from the command-line args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        "-o",
        type=pathlib.Path,
        required=True,
        help="checkpoint directory to write",
    )
    parser.add_argument(
        "--target-config",
        type=pathlib.Path,
        default=None,
        help="target config.json to read hidden size, vocab and layer count "
        "from when not given explicitly",
    )
    parser.add_argument("--target-hidden-size", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument(
        "--target-layer-ids",
        help="comma-separated 0-based post-layer taps, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="draft decoder layers",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    config_hidden: int | None = None
    config_vocab: int | None = None
    target_depth: int | None = None
    if args.target_config is not None and args.target_config.exists():
        config_hidden, config_vocab, target_depth = read_target_dims(
            args.target_config
        )
    if args.target_hidden_size is None or args.vocab_size is None:
        if config_hidden is None or config_vocab is None:
            parser.error(
                "no target dims: pass an existing --target-config, or "
                "--target-hidden-size and --vocab-size"
            )
        hidden = args.target_hidden_size or config_hidden
        vocab = args.vocab_size or config_vocab
    else:
        hidden = args.target_hidden_size
        vocab = args.vocab_size
    if args.target_layer_ids:
        taps = [int(t) for t in args.target_layer_ids.split(",")]
    elif target_depth is not None:
        taps = _default_taps(target_depth)
    else:
        parser.error(
            "no tap list: pass --target-layer-ids or a readable --target-config"
        )

    config = build_tiny_config(
        target_hidden_size=hidden,
        vocab_size=vocab,
        target_layer_ids=taps,
        target_num_hidden_layers=target_depth,
        num_layers=args.num_layers,
    )
    out = write_checkpoint(config, args.out, seed=args.seed)
    num_tensors = len(expected_state_dict_shapes(config))
    print(
        f"wrote {num_tensors} random-weight tensors to "
        f"{out / 'model.safetensors'} (seed={args.seed})"
    )


if __name__ == "__main__":
    main()
