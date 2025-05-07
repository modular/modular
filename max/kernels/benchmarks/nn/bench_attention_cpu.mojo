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
# RUN: %mojo-build-no-debug-no-assert %s

from os import abort
from random import rand

from benchmark import *
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from memory import UnsafePointer
from nn.flash_attention import flash_attention

from utils import IndexList
from utils.index import Index


@value
struct AttentionSpec(Stringable):
    var batch_size: Int
    var seq_len: Int
    var kv_seq_len: Int
    var depth_dim: Int

    @no_inline
    fn __str__(self) -> String:
        # fmt: off
        return String(
            "batch_size=", self.batch_size,
            ",seq_len=", self.seq_len,
            ",kv_seq_len=", self.kv_seq_len,
            ",depth_dim=", self.depth_dim,
        )
        # fmt: on


def bench_attention[type: DType](mut m: Bench, spec: AttentionSpec):
    var q_shape = Index(spec.batch_size, spec.seq_len, spec.depth_dim)
    var kv_shape = Index(spec.batch_size, spec.kv_seq_len, spec.depth_dim)
    var mask_shape = Index(spec.batch_size, spec.seq_len, spec.kv_seq_len)

    var q_ptr = UnsafePointer[Scalar[type]].alloc(q_shape.flattened_length())
    var k_ptr = UnsafePointer[Scalar[type]].alloc(kv_shape.flattened_length())
    var v_ptr = UnsafePointer[Scalar[type]].alloc(kv_shape.flattened_length())
    var mask_ptr = UnsafePointer[Scalar[type]].alloc(
        mask_shape.flattened_length()
    )
    var output_ptr = UnsafePointer[Scalar[type]].alloc(
        q_shape.flattened_length()
    )

    rand(q_ptr, q_shape.flattened_length())
    rand(k_ptr, kv_shape.flattened_length())
    rand(v_ptr, kv_shape.flattened_length())
    rand(mask_ptr, mask_shape.flattened_length())

    var q = NDBuffer[type, 3](q_ptr, q_shape)
    var k = NDBuffer[type, 3](k_ptr, kv_shape)
    var v = NDBuffer[type, 3](v_ptr, kv_shape)
    var mask = NDBuffer[type, 3](mask_ptr, mask_shape)
    var output = NDBuffer[type, 3](output_ptr, q_shape)

    @parameter
    @always_inline
    fn input_k_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return k.load[width=simd_width](rebind[IndexList[3]](idx))

    @parameter
    @always_inline
    fn input_v_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return v.load[width=simd_width](rebind[IndexList[3]](idx))

    @parameter
    @always_inline
    fn mask_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return mask.load[width=simd_width](rebind[IndexList[3]](idx))

    alias scale = 0.25

    @always_inline
    @parameter
    fn flash_bench_fn(mut b: Bencher):
        @always_inline
        @parameter
        fn iter_fn[depth_static_dim: Dim]():
            alias output_static_shape = DimList(Dim(), Dim(), depth_static_dim)
            flash_attention[input_k_fn, input_v_fn, mask_fn](
                q.make_dims_unknown(),
                k.get_shape(),
                v.get_shape(),
                mask.get_shape(),
                rebind[NDBuffer[type, 3, output.origin, output_static_shape]](
                    output
                ),
                scale=scale,
            )

        alias depth_static_dims = VariadicList[Int](40, 64, 80, 128, 160)

        @parameter
        for idx in range(len(depth_static_dims)):
            if depth_static_dims[idx] == spec.depth_dim:
                b.iter[iter_fn[Dim(depth_static_dims[idx])]]()
                return

        # Fallback to dispatch with a dynamic shape.
        b.iter[iter_fn[Dim()]]()

    m.bench_function[flash_bench_fn](BenchId("flash", String(spec)))

    _ = q
    _ = k
    _ = v
    _ = mask
    _ = output


def main():
    var specs = List[AttentionSpec](
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
    )

    var m = Bench()
    for i in range(len(specs)):
        bench_attention[DType.float32](m, specs[i])
    m.dump_report()
