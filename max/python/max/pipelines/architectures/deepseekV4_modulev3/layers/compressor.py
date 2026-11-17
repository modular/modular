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
"""The DeepSeek-V4 CSA compressor on ModuleV3, cached path only.

Stateless: the open window's raw ``wkv`` / ``wgate`` rows live in a state
leaf, and one kernel pools every window of the batch reading that leaf in
place. Overlap (ratio 4), ``ape`` and the fp32 pooling are described in
the ModuleV2 implementation (``deepseekV4/layers/compressor.py`` at
dfc8109fc0c).
"""

from __future__ import annotations

from max.dtype import DType
from max.nn import Linear, Module
from max.tensor import Tensor, default_dtype

from ..model_config import DeepseekV4Config
from .cache import CacheLeaf
from .kernels import compress_windows_ragged
from .norm import DeepseekV4RMSNorm
from .ragged import RaggedRows, window_coff

__all__ = ["DeepseekV4Compressor"]


class DeepseekV4Compressor(Module[..., Tensor]):
    """Gated pooling over ``compress_ratio`` consecutive tokens.

    ``rotate`` is the indexer's variant: Hadamard rotation and FP4 instead of
    FP8 on the pooled entry.
    """

    ape: Tensor

    def __init__(
        self,
        config: DeepseekV4Config,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
    ) -> None:
        super().__init__()
        self.rotate = rotate
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.coff = window_coff(compress_ratio)
        self.proj_dim = self.coff * head_dim
        self.ape = Tensor.zeros(
            [compress_ratio, self.proj_dim], dtype=DType.float32
        )
        # ``[wkv; wgate]`` stacked by the adapter: one f32 GEMM (GEX-1978).
        with default_dtype(DType.float32):
            self.wkv_wgate = Linear(
                config.hidden_size, 2 * self.proj_dim, bias=False
            )
        self.norm = DeepseekV4RMSNorm(head_dim, config.rms_norm_eps)
        self.activation_dtype = config.dtype

    def projections(self, x32: Tensor) -> tuple[Tensor, Tensor]:
        """Raw ``wkv`` / ``wgate`` rows, ``[T, proj_dim]`` float32 each."""
        out = self.wkv_wgate(x32)
        width = int(out.shape[-1])
        # Positive slice bounds: negative ones can come back rotated.
        return out[..., : self.proj_dim], out[..., self.proj_dim : width]

    def forward(
        self,
        kv_proj: Tensor,
        score_proj: Tensor,
        rows: RaggedRows,
        window_offsets: Tensor,
        freqs_table: Tensor,
        state: CacheLeaf,
        layer: Tensor,
    ) -> Tensor:
        """Every window of the batch pooled into one entry.

        Returns ``[n, head_dim]``, ``n`` a bound on the window count; rows
        past ``window_offsets[b]`` are left unwritten.
        """
        return compress_windows_ragged(
            kv_proj,
            score_proj,
            self.ape,
            # As the V2 ``RMSNorm`` applies it: the weight in the input dtype.
            self.norm.weight.cast(self.activation_dtype),
            freqs_table,
            rows.offsets,
            window_offsets,
            rows.starts,
            state.values,
            layer,
            epsilon=self.norm.eps,
            rotate=self.rotate,
        )
