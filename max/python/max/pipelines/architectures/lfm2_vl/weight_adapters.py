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
"""Checkpoint -> graph weight rules for Lfm2VlForConditionalGeneration.

The decoder graph takes the checkpoint's ``model.language_model.*`` tree
as ``language_model.*``; the vision graph takes
``model.vision_tower.vision_model.*`` at its bare tower names and
``model.multi_modal_projector.*`` as ``multi_modal_projector.*``.

Everything else is a fixed rename of the checkpoint's vendored layout
onto the modules' parameter names: the decoder's ``out_proj`` /
``q_layernorm`` / ``k_layernorm`` become ``o_proj`` / ``q_norm_weight`` /
``k_norm_weight``, the gated MLP's ``w1`` / ``w3`` / ``w2`` become
``gate_proj`` / ``up_proj`` / ``down_proj``, the final
``embedding_norm`` is the decoder's ``norm``, and the causal conv's
``conv.weight`` becomes the module's ``conv_weight`` (layers/conv.py
declares it in the shipped ``[hidden, 1, conv_L_cache]`` shape and
reshapes in-graph, so the rule is the rename alone).

Rules run in order, each on the previous rule's output; the three
prefix rules come first, so the decoder-scoped renames below them can
key on the ``language_model.`` prefix and leave the tower's attention
names untouched.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib import Ignore, WeightRenaming, WeightRule
from transformers.configuration_utils import PretrainedConfig


def lfm2vl_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules mapping this checkpoint onto its modules.

    Args:
        huggingface_config: The checkpoint's config; unused, since every
            rename is fixed by the published checkpoint layout.
    """
    del huggingface_config
    return (
        # Prefix split: decoder tree, bare vision tower tree, projector.
        WeightRenaming(r"^model\.language_model\.", "language_model."),
        WeightRenaming(r"^model\.vision_tower\.vision_model\.", ""),
        WeightRenaming(
            r"^model\.multi_modal_projector\.", "multi_modal_projector."
        ),
        # Decoder attention: vendored names onto the modules' names. The
        # vision tower's attention is also a ``self_attn`` with an
        # ``out_proj``, so these are scoped to the decoder prefix the
        # first rule put on; the tower keeps the reference's names.
        WeightRenaming(
            r"^(language_model\..*)\.self_attn\.out_proj\.weight$",
            r"\1.self_attn.o_proj.weight",
        ),
        WeightRenaming(
            r"^(language_model\..*)\.self_attn\.q_layernorm\.weight$",
            r"\1.self_attn.q_norm_weight",
        ),
        WeightRenaming(
            r"^(language_model\..*)\.self_attn\.k_layernorm\.weight$",
            r"\1.self_attn.k_norm_weight",
        ),
        # Causal conv kernel: rename only; the in-graph reshape lives in
        # layers/conv.py.
        WeightRenaming(r"\.conv\.conv\.weight$", ".conv.conv_weight"),
        # Final RMSNorm.
        WeightRenaming(r"\.embedding_norm\.weight$", ".norm.weight"),
        # Decoder gated MLP: SwiGLU vendored names onto module names.
        WeightRenaming(
            r"\.feed_forward\.w1\.weight$", ".feed_forward.gate_proj.weight"
        ),
        WeightRenaming(
            r"\.feed_forward\.w3\.weight$", ".feed_forward.up_proj.weight"
        ),
        WeightRenaming(
            r"\.feed_forward\.w2\.weight$", ".feed_forward.down_proj.weight"
        ),
        # A top-level head ships with an untied name; name the drop in
        # the load log rather than skip it silently.
        Ignore(r"^lm_head\.", reason="d1-3B ties lm_head to embed_tokens"),
    )
