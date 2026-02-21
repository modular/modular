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

from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_utils import (
    MHAConfig,
    OptionallyStaticInt,
    DynamicInt,
)
from nn.mha_mask import MHAMask
from gpu.host import DeviceContext
from layout.layout_tensor import LayoutTensor
from layout.layout import Layout
from gpu.memory import AddressSpace
from .mla_prefill_sm100_bf16 import mla_sm100_prefill_bf16
from .mla_prefill_sm100_fp8 import mla_sm100_prefill_fp8


@always_inline
fn mla_sm100_prefill[
    output_type: DType,
    q_type: DType,
    KVType: MHAOperand,
    KRopeType: MHAOperand,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    config: MHAConfig,
    group: Int,
    q_depth: Int,
    cache_depth: Int,
    use_score_mod: Bool,
    _ndbuffer_mha_operand: Bool,
](
    output: LayoutTensor[
        output_type, address_space = AddressSpace.GENERIC, ...
    ],
    q: LayoutTensor[q_type, _, address_space = AddressSpace.GENERIC, ...],
    k: KVType,
    v: KVType,
    k_rope: KRopeType,
    mask_functor: MaskType,
    score_mod_functor: ScoreModType,
    valid_length: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    max_prompt_len: MaxPromptLenType,
    scale: Float32,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    comptime if KRopeType.dtype == DType.bfloat16:
        mla_sm100_prefill_bf16[
            config=config,
            group = Int(group),
            q_depth=q_depth,
            cache_depth=cache_depth,
            use_score_mod=use_score_mod,
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
        ](
            output,
            q,
            k,
            rebind[type_of(k)](v),
            k_rope,
            mask_functor,
            score_mod_functor,
            valid_length,
            max_prompt_len,
            scale,
            batch_size,
            ctx,
        )
    else:
        mla_sm100_prefill_fp8[
            config=config,
            group = Int(group),
            q_depth=q_depth,
            cache_depth=cache_depth,
            use_score_mod=use_score_mod,
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
        ](
            output,
            q,
            k,
            rebind[type_of(k)](v),
            k_rope,
            mask_functor,
            score_mod_functor,
            valid_length,
            max_prompt_len,
            scale,
            batch_size,
            ctx,
        )
