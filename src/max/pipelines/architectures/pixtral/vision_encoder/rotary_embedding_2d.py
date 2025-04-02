# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


from dataclasses import dataclass
from functools import cached_property

from max.dtype import DType
from max.graph import Dim, DimLike, TensorValue, TensorValueLike, ops
from max.nn.layer import Layer


def meshgrid(height: DimLike, width: DimLike, indexing="ij") -> TensorValue:
    """Returns row indices and col indices of each point on the grid."""
    height = Dim(height)
    width = Dim(width)
    row_indices = ops.range(
        ops.constant(0, DType.int64),
        TensorValue(height),
        ops.constant(1, DType.int64),
        out_dim=height,
    )
    col_indices = ops.range(
        ops.constant(0, DType.int64),
        TensorValue(width),
        ops.constant(1, DType.int64),
        out_dim=width,
    )

    # repeat row indices for each row [[0, ..., 0], ..., [width=n_cols-1, ..., width-1]]
    h_grid = ops.tile(
        ops.unsqueeze(row_indices, 1), [1, width]
    )  # Shape: (height, width)
    # repeat col indices for each col [[0, 1, ..., height-1=n_rows-1], ...]
    v_grid = ops.tile(
        ops.unsqueeze(col_indices, 0), [height, 1]
    )  # Shape: (height, width)
    return h_grid, v_grid  # type: ignore


def patch_position_ids(
    patch_embeds: list[TensorValue], max_width: int
) -> TensorValue:
    """
    Takes a list of patches, calculates the positional indices for each patch by
    flattening the array, and returns these indices in the positions tensor.
    max_width is the maximum numbers of patches you can have in an image on one
    side ie.e max_image_width_or_height // patch_size.
    """
    positions = []
    for patch in patch_embeds:
        height, width = patch.shape[
            1:3
        ]  # img_height/patch_size, img_width/patch_size
        # TODO(MSDK-1194): replace with ops.meshgrid()
        mesh = meshgrid(height, width, indexing="ij")
        # TODO(MSDK-1193): replace ? by ops.chunk() or ops.split_tensor()
        # Combine row and col indices into 1 tensor of paired coordinates. Shape = (height, width, 2)
        # Then split into 2 tensors: 1st and 2nd coordinate of points in the mesh.
        mesh_coords = ops.stack(mesh, axis=-1).reshape((-1, 2))  # type: ignore
        h_grid, v_grid = mesh_coords[:, 0], mesh_coords[:, 1]
        # Calculates a unique ID for each coordinate pair.
        # TODO: Understand if using max_width here leads to memory inefficiency
        ids = h_grid * max_width + v_grid
        positions.append(ids[:])
    return ops.concat(positions)


@dataclass
class RotaryEmbedding2D(Layer):
    """
    RotaryEmbedding layer to calculate and apply the frequency tensor for complex exponentials.
    """

    dim: DimLike
    n_heads: int
    theta: float
    """Hyperparameter used to control the frequency scaling of the sinusoidal components of the embeddings."""
    max_patches_per_side: int
    """The maximum number of patches per side for model's input (images)."""

    def freqs_cis_base(self) -> TensorValue:
        """
        Computes the frequency tensor for complex exponentials (cis)
        for a given seq_len = num_patches in a block (image).
        Tensor is scaled with theta parameter.
        Required to apply Rotary Position Embedding (RoPE) to tensor.
        See 'Roformer: Enhanced Transformer with Rotary Embedding'
        (arxiv.org/pdf/2104.09864).

        Returns:
            The frequency tensor for complex exponentials with shape
                (max_seq_len, dim//(n_heads), 2)
                ??(batch_size, height * width, dim) with dim the embed dim.
        """
        head_dim = (
            self.dim // self.n_heads  # type: ignore
        )  # hidden_size // num_attention_heads
        # Note: using float64 to avoid an overflow on the exponential, then converting back to float32.
        # 1D tensor of length head_dim // 2 = 32
        iota = ops.range(
            ops.constant(0, DType.float64),
            ops.constant(head_dim, DType.float64),  # type: ignore
            ops.constant(2, DType.float64),
            out_dim=head_dim // 2,
        )
        # 1D tensor of length head_dim // 2 = 32
        freqs = ops.cast(1.0 / (self.theta ** (iota / head_dim)), DType.float32)

        # Indices of patches in each side (height and width) of image.
        # 1D tensor of length max_patches_per_side = 64
        h = ops.range(
            ops.constant(0, DType.float32),
            ops.constant(self.max_patches_per_side, DType.float32),
            ops.constant(1, DType.float32),
            out_dim=self.max_patches_per_side,
        )
        # 1D tensor of length max_patches_per_side = 64
        w = ops.range(
            ops.constant(0, DType.float32),
            ops.constant(self.max_patches_per_side, DType.float32),
            ops.constant(1, DType.float32),
            out_dim=self.max_patches_per_side,
        )
        # create matrices of freqs = outer product of height and width indices with their respective frequency.
        # 2D tensors mapping patch positions to rotary embeddings. shape =(max_patches_per_side = 64, head_dim//4 =16)
        freqs_h = ops.outer(h, freqs[::2])
        freqs_w = ops.outer(w, freqs[1::2])

        # Combines the frequency matrices of horizontal and vertical patch indices into a single tensor of patches
        # 2D tensor of shape (max_patches_per_side*max_patches_per_side = 4096,  head_dim // 2 = 32)
        _inv_freq = ops.concat(
            [
                ops.tile(
                    ops.unsqueeze(freqs_h, 1),
                    (1, self.max_patches_per_side, 1),
                ),
                ops.tile(
                    ops.unsqueeze(freqs_w, 0),
                    (self.max_patches_per_side, 1, 1),
                ),
            ],
            axis=-1,
        ).reshape((-1, head_dim // 2))

        # In Hugging Face Code, double copies to have rotary embeddings that match head_dim
        # 2D tensor of shape (max_patches_per_side*max_patches_per_side =4096, head_dim=64)
        _inv_freq = ops.concat((_inv_freq, _inv_freq), axis=-1)

        # 2D tensor of shape (max_patches_per_side*max_patches_per_side =4096, head_dim*2=128)
        # self._freqs_cis = ops.stack(
        #    [ops.cos(_inv_freq), ops.sin(_inv_freq)], axis=-1
        # )
        return TensorValue(_inv_freq)

    @cached_property
    def freqs_cis(self) -> TensorValue:
        self._freqs_cis = self.freqs_cis_base()
        return self._freqs_cis

    def __call__(
        self, x: TensorValueLike, position_ids: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Applies rotary positional embeddings (RoPE) to `x`.

        Args:
            x: Activation tensor with shape (batch, seq_len, n_kv_heads, head_dim).
            position_ids: starting position of input tensor

        Returns:
            Input activation tensor with rotary positional embeddings applied and
            the same shape as `x`.
        """
        v = TensorValue(x)

        # TODO: Figure out if this is the correct axis.
        freqs_cis_sliced = ops.gather(self.freqs_cis, position_ids, 0)
        # TODO(MSDK-1188): Ideally this cast would happen inside of the cached
        # self.freqs_cis property instead of here, but complex.dtype is not
        # known at that point.
        cos = ops.cast(ops.cos(freqs_cis_sliced), v.dtype)
        sin = ops.cast(ops.sin(freqs_cis_sliced), v.dtype)

        return cos, sin
