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

from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_utils import MHAConfig, OptionallyStaticInt
from nn.attention.mha_mask import MHAMask
from std.gpu.host import DeviceContext
from layout import TileTensor
from std.gpu.memory import AddressSpace
from .mla_prefill_generic import mla_sm100_prefill_generic
from .mla_prefill_blockscale import mla_sm100_prefill_blockscale


@always_inline
def mla_sm100_prefill[
    output_type: DType,
    q_type: DType,
    KVType: MHAOperand,
    KRopeType: MHAOperand,
    MaskType: MHAMask,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    config: MHAConfig,
    group: Int,
    q_depth: Int,
    cache_depth: Int,
    _ndbuffer_mha_operand: Bool,
    blockwise_scale: Int = 0,
](
    output: TileTensor[output_type, address_space=AddressSpace.GENERIC, ...],
    q: TileTensor[q_type, address_space=AddressSpace.GENERIC, ...],
    k: KVType,
    v: KVType,
    k_rope: KRopeType,
    mask_functor: MaskType,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    max_prompt_len: MaxPromptLenType,
    scale: Float32,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    comptime assert (
        output_type == DType.bfloat16
    ), "Only support bfloat16 output for SM100 MLA prefill"

    comptime if blockwise_scale == 0 and (
        KRopeType.dtype == KVType.dtype == q.dtype
    ):
        comptime assert (
            blockwise_scale == 0
        ), "blockwise_scale is not supported for generic MLA prefill"
        mla_sm100_prefill_generic[
            config=config,
            group=Int(group),
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
        ](
            output,
            q,
            k,
            rebind[type_of(k)](v),
            k_rope,
            mask_functor,
            valid_length,
            max_prompt_len,
            scale,
            batch_size,
            ctx,
        )
    else:
        mla_sm100_prefill_blockscale[
            config=config,
            group=Int(group),
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
            blockwise_scale=blockwise_scale,
        ](
            output,
            q,
            k,
            rebind[type_of(k)](v),
            k_rope,
            mask_functor,
            valid_length,
            max_prompt_len,
            scale,
            batch_size,
            ctx,
        )
