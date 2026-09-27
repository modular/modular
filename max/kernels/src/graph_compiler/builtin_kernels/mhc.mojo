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
"""Graph-op bindings for manifold-constrained hyper-connections (mHC).

The kernel math lives in `nn.mhc`. Registration must be declared inside the
built-in kernel library; importing the kernel from `nn` does not add the op to
the graph compiler's registry.
"""

import extensibility
from extensibility import InputTensor, OutputTensor
from max.gpu.host import DeviceContext

from nn.mhc import mhc_split_sinkhorn


@extensibility.register("mo.mhc.split_sinkhorn")
struct MHCSplitSinkhorn:
    """Registers the `mo.mhc.split_sinkhorn` graph op with the graph compiler.

    Splits per-token mHC mixing logits into the pre and post weights and the
    Sinkhorn-projected combination matrix. See `nn.mhc`.

    Tensor shapes:
        - pre    : [tokens, hc]                (OUT)
        - post   : [tokens, hc]                (OUT)
        - comb   : [tokens, hc * hc]           (OUT)
        - mixes  : [tokens, (2 + hc) * hc]
        - scale  : [3]
        - base   : [(2 + hc) * hc]
    """

    @inline(.always)
    @staticmethod
    def execute[
        hc_mult: Int,
        sinkhorn_iters: Int,
        target: StaticString,
        eps: StaticString = "1e-6",
        post_mult: StaticString = "2.0",
    ](
        pre: OutputTensor[dtype=.float32, rank=2, ...],
        post: OutputTensor[dtype=.float32, rank=2, ...],
        comb: OutputTensor[dtype=.float32, rank=2, ...],
        mixes: InputTensor[dtype=.float32, rank=2, ...],
        scale: InputTensor[dtype=.float32, rank=1, ...],
        base: InputTensor[dtype=.float32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        # `ops.custom` parameters cannot be floats, so the two constants
        # arrive string-encoded (the `attn_res_mix` convention).
        mhc_split_sinkhorn[hc_mult, target=target](
            pre.to_tile_tensor[.int64](),
            post.to_tile_tensor[.int64](),
            comb.to_tile_tensor[.int64](),
            mixes.to_tile_tensor[.int64](),
            scale.to_tile_tensor[.int64](),
            base.to_tile_tensor[.int64](),
            sinkhorn_iters,
            Float32(atof(eps)),
            Float32(atof(post_mult)),
            ctx,
        )
