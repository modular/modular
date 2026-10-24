//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Composite ops (`mo.composite.*`): dedicated ops for fused subgraphs.
// Pattern-based fusions replace a matched subgraph with one of these, never an
// `mo.custom`; they can also be emitted directly.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOCOMPOSITEOPS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOCOMPOSITEOPS_TD

include "GraphCompiler/MODialect/MOOps.td"

// Ops needing a more specialized base (e.g. `MO_DotBase`, `MO_DistributedOp`)
// instead spell out the `composite.` prefix, the `Composite` cppNamespace, and
// `MO_CompositeOpTrait` directly.
class MO_CompositeOp<string mnemonic, list<Trait> traits = []>
  : MO_TensorOp<"composite." # mnemonic, [MO_CompositeOpTrait] # traits> {
  string cppNamespace = MODialect.cppNamespace # "::Composite";
}

//===----------------------------------------------------------------------===//
// ReduceRMSNormRoPEOp
//===----------------------------------------------------------------------===//

def MOComposite_ReduceRMSNormRoPEOp : MO_CompositeOp<"rms_norm_rope", [
  Pure,
  MO_SingleDevice,
  MO_ParamDeclaration,
]> {
  let hasVerifier = 1;
  let summary = "Fused RMS normalization with Rotary Position Embedding";
  let description = [{
      normed = rms_norm(input, weight, epsilon, weight_offset)
      x1, x2 = split(normed, axis=-1)
      rotated = concat(-x2, x1, axis=-1)
      result = normed * cos_vals + rotated * sin_vals

    Example:

    ```mlir
      %result = mo.composite.rms_norm_rope(%input, %weight, %cos_vals,
                                           %sin_vals)
        {epsilon = #M.dense_array<9.99999997E-7> : tensor<f32>,
         weight_offset = #M.dense_array<0.0> : tensor<f32>,
         multiply_before_cast = false} :
        (!mo.tensor<[2, 3, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>,
         !mo.tensor<[2, 3, 128], f32, gpu:0>, !mo.tensor<[2, 3, 128], f32, gpu:0>)
        -> !mo.tensor<[2, 3, 128], bf16, gpu:0>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$weight,
    MO_Tensor:$cosVals,
    MO_Tensor:$sinVals,
    MO_F32ScalarAttr:$epsilon,
    MO_F32ScalarAttr:$weight_offset,
    BoolAttr:$multiply_before_cast,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `{` `epsilon` `=` $epsilon `,` `weight_offset` `=` $weight_offset `,`
    `multiply_before_cast` `=` $multiply_before_cast `}` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// RopeRaggedOp
//===----------------------------------------------------------------------===//

def MOComposite_RopeRaggedOp : MO_CompositeOp<"rope.ragged", [
  Pure,
  MO_SingleDevice,
  MO_ParamDeclaration,
]> {
  let summary = "Applies Rotary Position Embedding to ragged, cache-position-aware input";
  let description = [{
    Per-token positions come from `inputRowOffsets` (ragged batch boundaries)
    and `startPos` (each sequence's cache length) and index `freqsCis`. When
    `freqsCis` is narrower than `input`, only that many columns per head are
    rotated: the trailing ones by default (MLA layout), or the leading ones
    when `rope_first` is set (DeepSeekV3.2/GLM Indexer layout, `pe, nope`).
    `rope_first` must be false when `freqsCis` is as wide as `input`.

    Example:

    ```mlir
      %result = mo.composite.rope.ragged(%input, %row_offsets, %start_pos,
                                          %freqs_cis)
        {interleaved = false, rope_first = false} :
        (!mo.tensor<[8, 1, 64], bf16, gpu:0>, !mo.tensor<[batch_plus_one], ui32, gpu:0>,
         !mo.tensor<[batch], ui32, gpu:0>, !mo.tensor<[1024, 64], f32, gpu:0>)
        -> !mo.tensor<[8, 1, 64], bf16, gpu:0>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$inputRowOffsets,
    MO_Tensor:$startPos,
    MO_Tensor:$freqsCis,
    BoolAttr:$interleaved,
    BoolAttr:$rope_first,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `{` `interleaved` `=` $interleaved `,` `rope_first` `=` $rope_first `}`
    attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// LayerNormRopeRaggedOp
//===----------------------------------------------------------------------===//

def MOComposite_LayerNormRopeRaggedOp : MO_CompositeOp<"layer_norm_rope_ragged", [
  Pure,
  MO_SingleDeviceWithHostOperands<["epsilon"]>,
  MO_ParamDeclaration,
]> {
  let summary = "Fused LayerNorm with partial ragged Rotary Position Embedding";
  let description = [{
      normed = layer_norm(input, gamma, beta, epsilon)
      roped, passthrough = split(normed, axis=-1)
      roped = rope.ragged(roped, input_row_offsets, start_pos, freqs_cis)
      result = concat(roped, passthrough, axis=-1)

    The RoPE width is `freqsCis`'s last dimension.

    Example:

    ```mlir
      %result = mo.composite.layer_norm_rope_ragged(%input, %gamma, %beta,
                                                     %epsilon, %row_offsets,
                                                     %start_pos, %freqs_cis)
        {interleaved = false} :
        (!mo.tensor<[8, 128], bf16, gpu:0>, !mo.tensor<[128], f32, gpu:0>,
         !mo.tensor<[128], f32, gpu:0>, !mo.tensor<[], f32>,
         !mo.tensor<[batch_plus_one], ui32, gpu:0>, !mo.tensor<[batch], ui32, gpu:0>,
         !mo.tensor<[1024, 64], f32, gpu:0>)
        -> !mo.tensor<[8, 128], bf16, gpu:0>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$gamma,
    MO_Tensor:$beta,
    MO_Tensor:$epsilon,
    MO_Tensor:$inputRowOffsets,
    MO_Tensor:$startPos,
    MO_Tensor:$freqsCis,
    BoolAttr:$interleaved,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `{` `interleaved` `=` $interleaved `}` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// ReduceRMSNormFusedResidualAddOp
//===----------------------------------------------------------------------===//

def MOComposite_ReduceRMSNormFusedResidualAddOp : MO_CompositeOp<"rms_norm_fused_residual_add", [
  Pure,
  MO_SingleDevice,
  MO_ParamDeclaration,
]> {
  let summary = "Fused RMS norm with residual add operation";
  let description = [{
      intermediate = rms_norm(input, gamma1, epsilon1, weight_offset1) + residual_input
      output = rms_norm(intermediate, gamma2, epsilon2, weight_offset2)

    Example:

    ```mlir
      %output, %intermediate = mo.composite.rms_norm_fused_residual_add(
          %input, %residual, %gamma1, %gamma2) {
          epsilon1 = #M.dense_array<9.99999997E-7> : tensor<f32>, ...,
          multiply_before_cast = false} :
        (...) -> (!mo.tensor<[3, 2], f32>, !mo.tensor<[3, 2], f32>)
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$residualInput,
    MO_Tensor:$gamma1,
    MO_Tensor:$gamma2,
    MO_F32ScalarAttr:$epsilon1,
    MO_F32ScalarAttr:$epsilon2,
    MO_F32ScalarAttr:$weight_offset1,
    MO_F32ScalarAttr:$weight_offset2,
    BoolAttr:$multiply_before_cast,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$output, MO_Tensor:$intermediate);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `{` `epsilon1` `=` $epsilon1 `,` `epsilon2` `=` $epsilon2 `,`
    `weight_offset1` `=` $weight_offset1 `,`
    `weight_offset2` `=` $weight_offset2 `,`
    `multiply_before_cast` `=` $multiply_before_cast `}` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// RMSNormResidualAddOp
//===----------------------------------------------------------------------===//

def MOComposite_RMSNormResidualAddOp : MO_CompositeOp<"rms_norm_residual_add", [
  Pure,
  MO_SingleDevice,
  MO_ParamDeclaration,
]> {
  let summary = "Fused single-norm residual add + RMS norm operation";
  let description = [{
      intermediate = input + residual_input
      output = rms_norm(intermediate, gamma, epsilon, weight_offset)

    The transformer/mamba pre-norm boundary, where `intermediate` is the next
    block's residual. Unlike `rms_norm_fused_residual_add`, `input` is not
    normalized first.

    Example:

    ```mlir
      %output, %intermediate = mo.composite.rms_norm_residual_add(
          %input, %residual, %gamma) {
          epsilon = #M.dense_array<9.99999997E-7> : tensor<f32>,
          weight_offset = #M.dense_array<0.0> : tensor<f32>,
          multiply_before_cast = false} :
        (...) -> (!mo.tensor<[3, 2], f32>, !mo.tensor<[3, 2], f32>)
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$residualInput,
    MO_Tensor:$gamma,
    MO_F32ScalarAttr:$epsilon,
    MO_F32ScalarAttr:$weight_offset,
    BoolAttr:$multiply_before_cast,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$output, MO_Tensor:$intermediate);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `{` `epsilon` `=` $epsilon `,` `weight_offset` `=` $weight_offset `,`
    `multiply_before_cast` `=` $multiply_before_cast `}` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// FusedMatmulAddOp
//===----------------------------------------------------------------------===//

def MOComposite_FusedMatmulAddOp : MO_DotBase<"composite.matmul_add", [MO_CompositeOpTrait]> {
  string cppNamespace = MODialect.cppNamespace # "::Composite";
  let summary = "Fused matrix-matrix multiplication with element-wise add";
  let description = [{
    Computes C = A @ B (optionally transposed) + residual in one kernel.
    `residual` is rank-2 (output shape) or rank-1 (a bias broadcast along
    rows). Only lowered for SM100 (B200), as a matmul epilogue tensor.

    Example (2-D residual):

    ```mlir
      %res = mo.composite.matmul_add(%a, %b, %residual) [transpose_b = true] : (
        !mo.tensor<[4, 512], bf16>,
        !mo.tensor<[1536, 512], bf16>,
        !mo.tensor<[4, 1536], bf16>
      ) -> !mo.tensor<[4, 1536], bf16>
    ```

    Example (1-D bias broadcast):

    ```mlir
      %res = mo.composite.matmul_add(%a, %b, %bias) [transpose_b = true] : (
        !mo.tensor<[4, 512], bf16>,
        !mo.tensor<[1536, 512], bf16>,
        !mo.tensor<[1536], bf16>
      ) -> !mo.tensor<[4, 1536], bf16>
    ```
  }];

  let arguments = (
    ins MO_Rank2Tensor:$input_a,
    MO_Rank2Tensor:$input_b,
    MO_Tensor:$residual,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b
  );
  let results = (outs MO_Rank2Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)` (` ` `[` `transpose_b` `=` $transpose_b^ `]`)? attr-dict
    `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// AllReduceAddRMSNormQuantFP8Op
//===----------------------------------------------------------------------===//

def MOComposite_AllReduceAddRMSNormQuantFP8Op : MO_DistributedOp<"composite.distributed.allreduce_add_rms_norm_quant_fp8", [
  MO_MultiDevice,
  MO_CompositeOpTrait
]> {
  string cppNamespace = MODialect.cppNamespace # "::Composite";
  let summary = "Fused AllreduceSum + Add + RMSNorm + Quant across N devices for a single device.";
  let description = [{
    Sum-reduces same-shaped per-device inputs, adds the residual, then applies
    RMSNorm and dynamic FP8 quantization. Returns the quantized output, its
    scale, and the post-add residual.
  }];
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    Variadic<MO_Tensor>:$residual,
    Variadic<MO_Tensor>:$gamma,
    Variadic<MO_RankedScalarFloatTensor>: $scaleUb,
    MO_Chain:$inChain,
    MO_F32ScalarAttr:$epsilon,
    MO_F32ScalarAttr:$weight_offset
  );
  let results = (outs
    Variadic<MO_Tensor>:$output,
    Variadic<MO_Tensor>:$outScale,
    Variadic<MO_Tensor>:$outResidual,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)`
    `{` `epsilon` `=` $epsilon `,` `weight_offset` `=` $weight_offset `}`
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// BundledAllReduceAddRMSNormQuantFP8Op
//===----------------------------------------------------------------------===//

def MOComposite_BundledAllReduceAddRMSNormQuantFP8Op : MO_CompositeOp<
    "bundled.allreduce_add_rms_norm_quant_fp8", [
  MO_SingleDeviceWithPeerInputs<["scaleUb"]>,
  DeclareOpInterfaceMethods<MO_MutableOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  MO_SameVariadicOperandSize,
  MO_BundledCollective
]> {
  let summary = "Per-device fused allreduce.sum + add + rms_norm + fp8 "
                "quantize, for use inside mo.parallel.";
  let description = [{
    Bundled analog of
    `mo.composite.distributed.allreduce_add_rms_norm_quant_fp8`, used inside
    an `mo.parallel` region. Takes N peer inputs (from `mo.bundled.expand`)
    and N signal buffers (captured from graph scope).
  }];
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Tensor:$residual,
    MO_Tensor:$gamma,
    MO_RankedScalarFloatTensor:$scaleUb,
    MO_Chain:$inChain,
    MO_F32ScalarAttr:$epsilon,
    MO_F32ScalarAttr:$weight_offset
  );
  let results = (outs
    MO_Tensor:$output,
    MO_Tensor:$outScale,
    MO_Tensor:$outResidual,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)`
    `{` `epsilon` `=` $epsilon `,` `weight_offset` `=` $weight_offset `}`
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Flash attention ops
//===----------------------------------------------------------------------===//
//
// Each is a 1:1 wrapper around an existing flash-attention kernel. Splitting
// per (device, mask) keeps each kernel's signature and avoids dispatch in the
// graph compiler.

def MOComposite_NoMaskFlashAttentionCPUOp
    : MO_CompositeOp<"no_mask_flash_attention_cpu", [
  MO_SingleDeviceWithHostOperands<["scale"]>,
]> {
  let summary = "Fused multi-head (flash) attention on CPU, without a mask";
  let description = [{
    Fused scaled-dot-product attention (`softmax(Q @ K^T * scale) @ V`) on CPU
    with no attention mask. `scale` is a host scalar (`f32`). Lowers 1:1 to the
    `no_mask_flash_attention_cpu` kernel.
  }];
  let arguments = (ins
    MO_Tensor:$query,
    MO_Tensor:$key,
    MO_Tensor:$value,
    MO_Tensor:$scale
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOComposite_MaskedFlashAttentionCPUOp
    : MO_CompositeOp<"masked_flash_attention_cpu", [
  MO_SingleDeviceWithHostOperands<["scale"]>,
]> {
  let summary = "Fused multi-head (flash) attention on CPU, with a mask";
  let description = [{
    Fused scaled-dot-product attention (`softmax(Q @ K^T * scale + mask) @ V`)
    on CPU. `scale` is a host scalar (`f32`). Lowers 1:1 to the
    `with_mask_flash_attention_cpu` kernel.
  }];
  let arguments = (ins
    MO_Tensor:$query,
    MO_Tensor:$key,
    MO_Tensor:$value,
    MO_Tensor:$mask,
    MO_Tensor:$scale
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOComposite_MaskedFlashAttentionGPUOp
    : MO_CompositeOp<"masked_flash_attention_gpu", [
  MO_SingleDeviceWithHostOperands<["scale"]>,
]> {
  let summary = "Fused multi-head (flash) attention on GPU, with a mask";
  let description = [{
    Fused scaled-dot-product attention (`softmax(Q @ K^T * scale + mask) @ V`)
    on GPU. `scale` is a host scalar (`f32`). Lowers 1:1 to the
    `masked_flash_attention_gpu` kernel.
  }];
  let arguments = (ins
    MO_Tensor:$query,
    MO_Tensor:$key,
    MO_Tensor:$value,
    MO_Tensor:$mask,
    MO_Tensor:$scale
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// RMSNormFusedQuantizeDynamicScaledFP8Op
//===----------------------------------------------------------------------===//

def MOComposite_RMSNormFusedQuantizeDynamicScaledFP8Op
    : MO_CompositeOp<"rms_norm_fused_quantize_dynamic_scaled_fp8", [
  Pure,
  MO_SingleDeviceWithHostOperands<["scaleUb"]>,
]> {
  let summary = "Fused RMS norm + dynamic-scaled FP8 quantize";
  let description = [{
      normed = rms_norm(input, weight, epsilon, weight_offset)
                 {multiply_before_cast = true}
      output, scale = quantize_dynamic_scaled_float8(normed, scale_ub)

    `scale` is per-token. GPU only.
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$weight,
    MO_Tensor:$scaleUb,
    MO_F32ScalarAttr:$epsilon,
    MO_F32ScalarAttr:$weight_offset
  );
  let results = (outs MO_Tensor:$output, MO_Tensor:$scale);
  let assemblyFormat = [{
    `(` operands `)`
    `{` `epsilon` `=` $epsilon `,` `weight_offset` `=` $weight_offset `}`
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// MatmulFusedPartialRMSNormOp
//===----------------------------------------------------------------------===//

def MOComposite_MatmulFusedPartialRMSNormOp
    : MO_CompositeOp<"matmul_fused_partial_rms_norm", [
  Pure,
  MO_SingleDevice,
]> {
  let summary = "Fused matmul + partial RMS norm";
  let description = [{
      unnormed = input_a @ input_b   {transpose_b = true}
      normed   = rms_norm(unnormed, gamma, epsilon, weight_offset)
                   {multiply_before_cast = true}

    The kernel hardcodes both attributes above. GPU only.
  }];

  let arguments = (ins
    MO_Rank2Tensor:$input_a,
    MO_Rank2Tensor:$input_b,
    MO_Tensor:$gamma,
    MO_F32ScalarAttr:$epsilon,
    MO_F32ScalarAttr:$weight_offset
  );
  let results = (outs MO_Tensor:$normed, MO_Tensor:$unnormed);
  let assemblyFormat = [{
    `(` operands `)`
    `{` `epsilon` `=` $epsilon `,` `weight_offset` `=` $weight_offset `}`
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// DistributedMatmulReduceScatterSumOp
//===----------------------------------------------------------------------===//

def MOComposite_DistributedMatmulReduceScatterSumOp : MO_DistributedOp<"composite.distributed.matmul_reduce_scatter.sum", [
  MO_MultiDeviceFrom<"inputsA">,
  MO_CompositeOpTrait
]> {
  string cppNamespace = MODialect.cppNamespace # "::Composite";
  let summary = "Fused Matmul + ReduceScatter sum across N devices.";
  let description = [{
    Each device computes A_i @ B_i^T and the results are reduce-scattered
    along M. On Blackwell this is a fused kernel overlapping compute with
    cross-GPU TMA reductions; elsewhere it falls back to separate ops.

    All A inputs are rank-2 with one shape, likewise all B inputs. B is always
    transposed (K-major).

    When `has_residual` is true, `residual` is added only on peer
    `residual_peer` before communication, so the sum yields
    `sum_j(A_j @ B_j) + residual` (the TP attention
    `add(residual, matmul) -> reduce_scatter`, e.g. DeepseekV3/KimiK2.5 with
    TP+EP). Otherwise `residual` and `residual_peer` are ignored.
  }];
  let hasVerifier = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputsA,
    Variadic<MO_Tensor>:$inputsB,
    MO_Tensor:$residual,
    DefaultValuedAttr<BoolAttr, "false">:$has_residual,
    DefaultValuedAttr<I64Attr, "0">:$residual_peer,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Chain:$inChain
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputs,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)`
    `` custom<MatmulReduceScatterSumAttrs>($has_residual, $residual_peer)
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// DistributedReduceScatterRMSNormOp
//===----------------------------------------------------------------------===//

def MOComposite_DistributedReduceScatterRMSNormOp : MO_DistributedOp<"composite.distributed.reduce_scatter_rms_norm", [
  MO_MultiDevice,
  MO_CompositeOpTrait
]> {
  string cppNamespace = MODialect.cppNamespace # "::Composite";
  let summary = "Fused ReduceScatter sum + RMSNorm across N devices for a single device.";
  let description = [{
    Each device receives a disjoint row shard of the sum, kept in f32 and
    RMSNorm-normalized in the same launch (no separate-norm HBM round-trip).
    Returns the normed shard (`output`) and the sum shard (`outResidual`, the
    residual stream). The norm is `multiply_before_cast=true`. bf16 in/out
    only.

    When `has_residual` is true, each device adds its row shard of
    `residuals` in f32 before the norm. `residuals` must be REPLICATED
    (bit-identical on every rank of a group) so the per-rank add reproduces
    the leader-side pre-add it replaces. When false, the operands stay
    present and group-sized (variadic groups must match), filled with the
    inputs and never read, as with `has_residual` on
    `mo.composite.distributed.matmul_reduce_scatter.sum`.

    `group_size` matches `mo.distributed.reducescatter.sum`: devices split
    into contiguous groups that reduce independently (TP-within-DP). It must
    be at least 2 and divide the device count. The builder always sets it;
    the `0` default is not usable.
  }];
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    Variadic<MO_Tensor>:$gamma,
    Variadic<MO_Tensor>:$residuals,
    MO_Chain:$inChain,
    MO_F32ScalarAttr:$epsilon,
    MO_F32ScalarAttr:$weight_offset,
    DefaultValuedAttr<I64Attr, "0">:$group_size,
    DefaultValuedAttr<BoolAttr, "false">:$has_residual
  );
  let results = (outs
    Variadic<MO_Tensor>:$output,
    Variadic<MO_Tensor>:$outResidual,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)`
    `{` `epsilon` `=` $epsilon `,` `weight_offset` `=` $weight_offset `}`
    `` custom<ReduceScatterRMSNormAttrs>($group_size, $has_residual)
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// DistributedAllGatherRMSNormOp
//===----------------------------------------------------------------------===//

def MOComposite_DistributedAllGatherRMSNormOp : MO_DistributedOp<"composite.distributed.allgather_rms_norm", [
  MO_MultiDevice,
  MO_CompositeOpTrait
]> {
  string cppNamespace = MODialect.cppNamespace # "::Composite";
  let summary = "Fused AllGather + RMSNorm across N devices for a single device.";
  let description = [{
    Gathers the per-device row shards and RMSNorms the result in the same
    launch. Returns the normed tensor (`output`) and the raw gathered residual
    (`outResidual`), bit-identical to a standalone all-gather.
    `multiply_before_cast=true`, bf16 in/out only.

    `group_size` matches `mo.distributed.allgather`: devices split into
    contiguous groups that gather independently (TP-within-DP), so each output
    is its group's tensor. It must be at least 2 and divide the device count.
    The builder always sets it; the `0` default is not usable.
  }];
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    Variadic<MO_Tensor>:$gamma,
    MO_Chain:$inChain,
    MO_F32ScalarAttr:$epsilon,
    MO_F32ScalarAttr:$weight_offset,
    DefaultValuedAttr<I64Attr, "0">:$group_size
  );
  let results = (outs
    Variadic<MO_Tensor>:$output,
    Variadic<MO_Tensor>:$outResidual,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)`
    `{` `epsilon` `=` $epsilon `,` `weight_offset` `=` $weight_offset `}`
    (` ` `[` `group_size` `=` $group_size^ `]`)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def MOComposite_DistributedAllGatherRMSNormQuantMXFP8Op : MO_DistributedOp<"composite.distributed.allgather_rms_norm_quant_mxfp8", [
  MO_MultiDevice,
  MO_CompositeOpTrait
]> {
  string cppNamespace = MODialect.cppNamespace # "::Composite";
  let summary = "Fused AllGather + RMSNorm + MXFP8 quant across N devices for a single device.";
  let description = [{
    `mo.composite.distributed.allgather_rms_norm` plus an MXFP8 copy of the
    normed `output`: `outQuant` (float8_e4m3fn, same shape) and `outScale`
    (float8_e8m0fnu, `[rows, cols / 32]`, plain rank-2 row-major -- what
    `block_scaled_matmul_amd` takes as `a_scales`, NOT the SM100 SF-atom
    interleave and NOT the preshuffled atom order `block_scaled_matmul_amd_preb`
    requires). Quantized from the bf16 written to `output`, so byte-identical
    to a standalone quantize. Same `group_size` contract.
  }];
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    Variadic<MO_Tensor>:$gamma,
    MO_Chain:$inChain,
    MO_F32ScalarAttr:$epsilon,
    MO_F32ScalarAttr:$weight_offset,
    DefaultValuedAttr<I64Attr, "0">:$group_size
  );
  let results = (outs
    Variadic<MO_Tensor>:$output,
    Variadic<MO_Tensor>:$outQuant,
    Variadic<MO_Tensor>:$outScale,
    Variadic<MO_Tensor>:$outResidual,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)`
    `{` `epsilon` `=` $epsilon `,` `weight_offset` `=` $weight_offset `}`
    (` ` `[` `group_size` `=` $group_size^ `]`)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def MOComposite_DistributedAllGatherRMSNormQuantMXFP6Op : MO_DistributedOp<"composite.distributed.allgather_rms_norm_quant_mxfp6", [
  MO_MultiDevice,
  MO_CompositeOpTrait
]> {
  string cppNamespace = MODialect.cppNamespace # "::Composite";
  let summary = "Fused AllGather + RMSNorm + MXFP6 quant across N devices for a single device.";
  let description = [{
    The MXFP6 sibling of
    `mo.composite.distributed.allgather_rms_norm_quant_mxfp8`, a separate op
    because the result shape differs: four six-bit codes pack into three
    bytes, so `outQuant` is `ui8` `[rows, cols * 3 / 4]`. `outScale` and the
    `group_size` contract match the MXFP8 op.

    `fp6_format` selects the OCP encoding, 0 for E2M3 and 1 for E3M2. It must
    be explicit because the packed `ui8` result type cannot record it.
  }];
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    Variadic<MO_Tensor>:$gamma,
    MO_Chain:$inChain,
    MO_F32ScalarAttr:$epsilon,
    MO_F32ScalarAttr:$weight_offset,
    DefaultValuedAttr<I64Attr, "0">:$group_size,
    DefaultValuedAttr<I64Attr, "0">:$fp6_format
  );
  let results = (outs
    Variadic<MO_Tensor>:$output,
    Variadic<MO_Tensor>:$outQuant,
    Variadic<MO_Tensor>:$outScale,
    Variadic<MO_Tensor>:$outResidual,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)`
    `{` `epsilon` `=` $epsilon `,` `weight_offset` `=` $weight_offset `}`
    `` custom<AllGatherRMSNormQuantMXFP6Attrs>($group_size, $fp6_format)
    attr-dict `:` functional-type(operands, results)
  }];
}

// Grouped-matmul MoE FFN composite ops (NVFP4 expert-parallel path)
//===----------------------------------------------------------------------===//
//
// The two legs of an NVFP4 MoE FFN: gate-up GEMM + SwiGLU + bf16->nvfp4 quant
// (L1) and down GEMM (L2). The model graph emits these typed ops so the MegaFFN
// fusion below can match named operands. Each lowers 1:1 to its kernel.
//
// Routing operands and expert-usage scalars come from the EP gate on-device;
// the kernel's host contract (front-packed active experts,
// `num_experts <= 128`, `ids[e] >= 0 => tokens[e] > 0`) is an upstream
// guarantee, not verified here.

def MOComposite_GroupedMatmulSwigluNvfp4Op
    : MO_CompositeOp<"grouped_matmul_swiglu_nvfp4", [
  Pure,
  MO_SingleDeviceWithHostOperands<["estimated_total_m", "num_active_experts"]>,
]> {
  let summary = "Grouped gate-up GEMM + SwiGLU + bf16->nvfp4 quantize (MoE L1)";
  let description = [{
    Per-expert (ragged) grouped matmul of packed-NVFP4 activations by the
    gate-up weights, then SwiGLU and bf16->nvfp4 quantization. Returns the
    packed activations for the down leg (`c_packed`) and their per-expert
    SwiGLU scale tile (`c_swiglu_scales`).

    The gated activation is built at compile time from the `activation`,
    `swiglu_alpha` and `swiglu_limit` attributes.

    When `has_a_row_scales` is set, `a_row_scales` holds one bf16 scale per
    `hidden_states` row, applied with its expert scale before the SwiGLU.
    Otherwise it is an unread placeholder.
  }];

  let arguments = (ins
    MO_Tensor:$hidden_states,
    MO_Tensor:$weight,
    MO_Tensor:$a_scales,
    MO_Tensor:$b_scales,
    MO_Tensor:$expert_start_indices,
    MO_Tensor:$expert_ids,
    MO_Tensor:$a_scale_offsets,
    MO_Tensor:$expert_scales,
    MO_Tensor:$a_row_scales,
    MO_Tensor:$c_input_scales,
    MO_Tensor:$estimated_total_m,
    MO_Tensor:$num_active_experts,
    DefaultValuedAttr<StrAttr, "\"silu\"">:$activation,
    MO_F32ScalarAttr:$swiglu_alpha,
    MO_F32ScalarAttr:$swiglu_limit,
    DefaultValuedAttr<BoolAttr, "false">:$has_a_row_scales
  );
  let results = (outs MO_Tensor:$c_packed, MO_Tensor:$c_swiglu_scales);
  let assemblyFormat = [{
    `(` operands `)`
    `` custom<GroupedMatmulSwigluNvfp4Attrs>($activation, $swiglu_alpha,
                                             $swiglu_limit, $has_a_row_scales)
    attr-dict `:` functional-type(operands, results)
  }];
}

def MOComposite_GroupedMatmulBlockScaledOp
    : MO_CompositeOp<"grouped_matmul_block_scaled", [
  Pure,
  MO_SingleDeviceWithHostOperands<["estimated_total_m", "num_active_experts"]>,
]> {
  let summary = "Grouped block-scaled down GEMM (MoE L2)";
  let description = [{
    Per-expert (ragged) block-scaled grouped matmul of packed-NVFP4
    activations by the down weights, producing bf16 output.

    When `has_a_row_scales` is set, `a_row_scales` holds one bf16 scale per
    `hidden_states` row, applied to that output row with its expert scale.
    Otherwise it is an unread placeholder.
  }];

  let arguments = (ins
    MO_Tensor:$hidden_states,
    MO_Tensor:$weight,
    MO_Tensor:$a_scales,
    MO_Tensor:$b_scales,
    MO_Tensor:$expert_start_indices,
    MO_Tensor:$expert_ids,
    MO_Tensor:$a_scale_offsets,
    MO_Tensor:$expert_scales,
    MO_Tensor:$a_row_scales,
    MO_Tensor:$estimated_total_m,
    MO_Tensor:$num_active_experts,
    DefaultValuedAttr<BoolAttr, "false">:$has_a_row_scales
  );
  let results = (outs MO_Tensor:$output);
  let assemblyFormat = [{
    `(` operands `)`
    (` ` `[` `has_a_row_scales` `=` $has_a_row_scales^ `]`)? attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// MegaFFNNvfp4Op
//===----------------------------------------------------------------------===//

def MOComposite_MegaFFNNvfp4Op : MO_CompositeOp<"mega_ffn_nvfp4", [
  // Not `Pure`: writes `arrival_count` in place. `getEffects` is hand-written
  // in MOOps.cpp.
  DeclareOpInterfaceMethods<MO_MutableOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  MO_SingleDeviceWithHostOperands<[
    "estimated_total_m", "gate_up_num_active_experts",
    "down_num_active_experts"]>,
]> {
  let summary = "Fused NVFP4 MoE FFN (gate-up + SwiGLU + quant + down) in one "
                "kernel";
  let description = [{
    Collapses `mo.composite.grouped_matmul_swiglu_nvfp4` and
    `mo.composite.grouped_matmul_block_scaled` into one launch with an
    in-kernel cross-CTA hand-off. The intermediate activations and SwiGLU
    scales stay on-chip. B200 (SM100) only; elsewhere the unfused chain is
    kept. Operands are the union of both legs' inputs minus the intermediate.

    `arrival_count` holds the cross-CTA pool-slot arrival counters. It is a
    PERSISTENT buffer (`mo.buffer.create` with zero `initValue`) zeroed once at
    model load. Under `POST_SELF_CLEAN_UP` each launch accumulates into the
    half of each slot its generation parity selects and clears the half its
    predecessor used, so it is NOT all-zero between launches. Nothing may
    reset it per launch: that pins the generation and disarms the rotation.
    `inChain`/`outChain` sequence the write and are dropped before kernel-arg
    binding, so `arrival_count` binds to the `MutableInputTensor` arg.

    The gate-up leg's gated activation rides on the discardable `activation`,
    `swiglu_alpha` and `swiglu_limit` attributes, copied from
    `mo.composite.grouped_matmul_swiglu_nvfp4` by the fusion.

    `gate_up_a_row_scales`/`has_gate_up_a_row_scales` are the gate-up leg's
    per-row input scales (see `mo.composite.grouped_matmul_swiglu_nvfp4`).
  }];

  let arguments = (ins
    MO_Tensor:$hidden_states,
    MO_Tensor:$gate_up_weight,
    MO_Tensor:$gate_up_a_scales,
    MO_Tensor:$gate_up_b_scales,
    MO_Tensor:$down_weight,
    MO_Tensor:$down_b_scales,
    MO_Tensor:$expert_start_indices,
    MO_Tensor:$expert_ids,
    MO_Tensor:$a_scale_offsets,
    MO_Tensor:$gate_up_expert_scales,
    MO_Tensor:$gate_up_a_row_scales,
    MO_Tensor:$down_expert_scales,
    MO_Tensor:$c_input_scales,
    MO_Tensor:$estimated_total_m,
    MO_Tensor:$gate_up_num_active_experts,
    MO_Tensor:$down_num_active_experts,
    MO_Buffer:$arrival_count,
    MO_Chain:$inChain,
    DefaultValuedAttr<BoolAttr, "false">:$has_gate_up_a_row_scales
  );
  let results = (outs MO_Tensor:$output, MO_Chain:$outChain);
  let assemblyFormat = [{
    `(` operands `)`
    (` ` `[` `has_gate_up_a_row_scales` `=` $has_gate_up_a_row_scales^ `]`)?
    attr-dict `:` functional-type(operands, results)
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOCOMPOSITEOPS_TD
