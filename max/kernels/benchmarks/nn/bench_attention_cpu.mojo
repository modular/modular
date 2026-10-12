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

from std.random import rand

from std.benchmark import *
from std.memory import alloc, dealloc
from std.sys.info import align_of
from layout import Coord, TileTensor, UNKNOWN_VALUE, row_major
from nn.attention.cpu.mha import flash_attention

from std.utils import IndexList
from std.utils.index import Index


@fieldwise_init
struct AttentionSpec(ImplicitlyCopyable, Writable):
    var batch_size: Int
    var seq_len: Int
    var kv_seq_len: Int
    var depth_dim: Int

    # fmt: off
    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of the attention spec.

        Args:
            writer: The writer to write to.
        """
        writer.write(
            "batch_size=", self.batch_size,
            ",seq_len=", self.seq_len,
            ",kv_seq_len=", self.kv_seq_len,
            ",depth_dim=", self.depth_dim,
        )
    # fmt: on


def bench_attention[dtype: DType](mut m: Bench, spec: AttentionSpec) raises:
    var q_shape = Index(spec.batch_size, spec.seq_len, spec.depth_dim)
    var kv_shape = Index(spec.batch_size, spec.kv_seq_len, spec.depth_dim)
    var mask_shape = Index(spec.batch_size, spec.seq_len, spec.kv_seq_len)

    var q_alloc = alloc[Scalar[dtype]](
        {count = q_shape.flattened_length()}
    ).into_managed()
    var k_alloc = alloc[Scalar[dtype]](
        {count = kv_shape.flattened_length()}
    ).into_managed()
    var v_alloc = alloc[Scalar[dtype]](
        {count = kv_shape.flattened_length()}
    ).into_managed()
    var mask_alloc = alloc[Scalar[dtype]](
        {count = mask_shape.flattened_length()}
    ).into_managed()
    var output_alloc = alloc[Scalar[dtype]](
        {count = q_shape.flattened_length()}
    ).into_managed()

    rand(q_alloc.unsafe_span())
    rand(k_alloc.unsafe_span())
    rand(v_alloc.unsafe_span())
    rand(mask_alloc.unsafe_span())

    var q = (
        TileTensor(q_alloc.unsafe_span(), row_major(q_alloc.layout().count()))
        .reshape(Coord(q_shape))
        .as_imm()
    )
    var k = (
        TileTensor(k_alloc.unsafe_span(), row_major(k_alloc.layout().count()))
        .reshape(Coord(kv_shape))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var v = (
        TileTensor(v_alloc.unsafe_span(), row_major(v_alloc.layout().count()))
        .reshape(Coord(kv_shape))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var mask = (
        TileTensor(
            mask_alloc.unsafe_span(), row_major(mask_alloc.layout().count())
        )
        .reshape(Coord(mask_shape))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var output = TileTensor(
        output_alloc.unsafe_span(), row_major(output_alloc.layout().count())
    ).reshape(Coord(q_shape))

    @inline(.always)
    def input_k_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) capturing -> SIMD[dtype, simd_width]:
        comptime assert _rank == 3
        return k.load[width=simd_width, alignment=align_of[dtype]()](Coord(idx))

    @inline(.always)
    def input_v_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) capturing -> SIMD[dtype, simd_width]:
        comptime assert _rank == 3
        return v.load[width=simd_width, alignment=align_of[dtype]()](Coord(idx))

    @inline(.always)
    def mask_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) capturing -> SIMD[dtype, simd_width]:
        comptime assert _rank == 3
        return mask.load[width=simd_width, alignment=align_of[dtype]()](
            Coord(idx)
        )

    comptime scale = 0.25

    @inline(.always)
    def flash_bench_fn(mut b: Bencher) {imm}:
        @inline(.always)
        def iter_fn[depth_static_dim: Int]() {imm}:
            flash_attention[input_k_fn, input_v_fn, mask_fn](
                q,
                kv_shape,
                kv_shape,
                mask_shape,
                output,
                scale=scale,
            )

        comptime depth_static_dims = [40, 64, 80, 128, 160]

        comptime for idx in range(len(depth_static_dims)):
            comptime dim = depth_static_dims[idx]
            if dim == spec.depth_dim:
                # `iter` takes a closure value, and a parametric closure only
                # names an overload set, so instantiate it behind a
                # non-parametric one.
                @inline(.always)
                def iter_static() {imm}:
                    iter_fn[dim]()

                b.iter(iter_static)
                return

        # Benchmark any remaining depth through the same runtime-shape entry point.
        @inline(.always)
        def iter_dynamic() {imm}:
            iter_fn[UNKNOWN_VALUE]()

        b.iter(iter_dynamic)

    m.bench_function(flash_bench_fn, BenchId("flash", String(spec)))

    dealloc(q_alloc^)
    dealloc(k_alloc^)
    dealloc(v_alloc^)
    dealloc(mask_alloc^)
    dealloc(output_alloc^)


def main() raises:
    var specs = [
        # bert-base-uncased-seqlen-16-onnx.yaml
        AttentionSpec(
            batch_size=12,
            seq_len=16,
            kv_seq_len=16,
            depth_dim=64,
        ),
        # BERT/bert-base-uncased-seqlen-128-onnx.yaml
        # GPT-2/gpt2-small-seqlen-128.yaml
        # RoBERTa/roberta-base-hf-onnx.yaml
        AttentionSpec(
            batch_size=12,
            seq_len=128,
            kv_seq_len=128,
            depth_dim=64,
        ),
        # CLIP-ViT/clip-vit-large-patch14-onnx.yaml
        AttentionSpec(
            batch_size=16,
            seq_len=257,
            kv_seq_len=257,
            depth_dim=64,
        ),
        # Llama2/llama2-7B-MS-context-encoding-onnx.yaml
        AttentionSpec(
            batch_size=32,
            seq_len=100,
            kv_seq_len=100,
            depth_dim=128,
        ),
        # Llama2/llama2-7B-MS-token-gen-onnx.yaml
        # Mistral/mistral-7b-hf-onnx-LPTG.yaml
        AttentionSpec(
            batch_size=32,
            seq_len=1,
            kv_seq_len=1025,
            depth_dim=128,
        ),
        # Mistral/mistral-7b-hf-onnx-context-encoding-onnx.yaml
        AttentionSpec(
            batch_size=32,
            seq_len=1024,
            kv_seq_len=1024,
            depth_dim=128,
        ),
        # OpenCLIP/clip-dynamic-per-tensor-weight-type-quint8-onnx-optimized.yaml
        AttentionSpec(
            batch_size=12,
            seq_len=50,
            kv_seq_len=50,
            depth_dim=64,
        ),
        AttentionSpec(
            batch_size=24,
            seq_len=77,
            kv_seq_len=77,
            depth_dim=64,
        ),
        # ReplitV1.5/replitv15-3B-hf-context-encoding-onnx.yaml
        AttentionSpec(
            batch_size=24,
            seq_len=1024,
            kv_seq_len=1024,
            depth_dim=128,
        ),
        # ReplitV1.5/replitv15-3B-hf-LPTG-onnx.yaml
        AttentionSpec(
            batch_size=24,
            seq_len=1,
            kv_seq_len=1025,
            depth_dim=128,
        ),
        # StableDiffusion-1.x/text_encoder/text_encoder-onnx.yaml
        AttentionSpec(
            batch_size=24,
            seq_len=16,
            kv_seq_len=16,
            depth_dim=64,
        ),
        # StableDiffusion-1.x/unet/unet-onnx.yaml
        AttentionSpec(
            batch_size=16,
            seq_len=64,
            kv_seq_len=16,
            depth_dim=160,
        ),
        AttentionSpec(
            batch_size=16,
            seq_len=64,
            kv_seq_len=64,
            depth_dim=160,
        ),
        AttentionSpec(
            batch_size=16,
            seq_len=256,
            kv_seq_len=16,
            depth_dim=160,
        ),
        AttentionSpec(
            batch_size=16,
            seq_len=256,
            kv_seq_len=256,
            depth_dim=160,
        ),
        AttentionSpec(
            batch_size=16,
            seq_len=1024,
            kv_seq_len=16,
            depth_dim=80,
        ),
        AttentionSpec(
            batch_size=16,
            seq_len=1024,
            kv_seq_len=1024,
            depth_dim=80,
        ),
        AttentionSpec(
            batch_size=16,
            seq_len=4096,
            kv_seq_len=16,
            depth_dim=40,
        ),
        AttentionSpec(
            batch_size=16,
            seq_len=4096,
            kv_seq_len=4096,
            depth_dim=40,
        ),
        # StableDiffusion-1.x/vae_decoder/vae_decoder-onnx.yaml
        # StableDiffusion-1.x/vae_encoder/vae_encoder-onnx.yaml
        AttentionSpec(
            batch_size=2,
            seq_len=4096,
            kv_seq_len=4096,
            depth_dim=512,
        ),
        # StarCoder/starcoder-7b-hf-context-encoding-onnx.yaml
        AttentionSpec(
            batch_size=1,
            seq_len=32768,
            kv_seq_len=1024,
            depth_dim=128,
        ),
        # StarCoder/starcoder-7b-hf-token-gen-onnx.yaml
        AttentionSpec(
            batch_size=12,
            seq_len=16,
            kv_seq_len=16,
            depth_dim=64,
        ),
        # WavLM/wavlm-large-onnx.yaml
        AttentionSpec(
            batch_size=32,
            seq_len=49,
            kv_seq_len=49,
            depth_dim=64,
        ),
        # Whisper/decoder_model_merged/decoder_model_merged-onnx.yaml
        AttentionSpec(
            batch_size=16,
            seq_len=1,
            kv_seq_len=16,
            depth_dim=64,
        ),
        # Whisper/encoder_model/encoder_model-onnx.yaml
        AttentionSpec(
            batch_size=8,
            seq_len=1500,
            kv_seq_len=1500,
            depth_dim=64,
        ),
    ]

    var m = Bench()
    for i in range(len(specs)):
        bench_attention[.float32](m, specs[i])
    m.dump_report()
