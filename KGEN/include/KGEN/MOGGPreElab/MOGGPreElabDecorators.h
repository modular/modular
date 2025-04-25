//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOGGPREELAB_MOGGDECORATORS_H
#define KGEN_LIB_MOGGPREELAB_MOGGDECORATORS_H

#include "KGEN/KGENDialect/KGENUtils.h"
#include "llvm/ADT/StringRef.h"

//===----------------------------------------------------------------------===//
// Deprecated Tensor API definitions (will be removed)
//===----------------------------------------------------------------------===//

namespace M::KGEN::MOGGPreElab {

// Attribute on generator ops to look for which marks the function as being a
// kernel.
constexpr StringLiteral kernelRegistrationAttr = "mogg.kernel";

inline bool isKernel(Operation *gen) {
  return gen != nullptr && gen->hasAttr(kernelRegistrationAttr);
}

constexpr StringLiteral shapeFuncRegistrationAttr = "mogg.v1_shape_func";

constexpr StringLiteral IS_VIEW_ATTR = "mogg.view";
constexpr StringLiteral OUTLINED_ATTR = "mogg.outlined";

/// Tracks the mojo parameter value for each of the input parameters.
constexpr StringLiteral MOGG_ARG_PARAMS = "mogg.arg_params";
constexpr StringLiteral MOGG_RESULT_PARAMS = "mogg.result_params";
constexpr StringLiteral MOGG_ARG_TYPE_NAMES = "mogg.arg_type_names";
constexpr StringLiteral MOGG_RESULT_TYPE_NAME = "mogg.result_type_name";
constexpr StringLiteral MOGG_INPUT_PARAM_TYPES = "mogg.input_param_types";

// The names as they appear in the lit source.
constexpr StringLiteral MOGG_ARG_SRC_NAMES = "mogg.arg_src_names";

/// Tracks the mojo trait conformances of each argument and result type.
constexpr StringLiteral MOGG_ARGUMENT_CONFORMANCES = "mogg.arg_conformances";
constexpr StringLiteral MOGG_RESULT_CONFORMANCES = "mogg.result_conformances";

/// MOGG Intrinsic for the register kernel decorator.
constexpr StringLiteral MOGG_INTRINSIC_REGISTER = "mogg.intrinsic_register";

/// MOGG Intrinsic for the view kernel decorator.
constexpr StringLiteral MOGG_INTRINSIC_VIEW_KERNEL = "mogg.view_kernel";

/// MOGG Intrinsic for the for_each function
constexpr StringLiteral MOGG_INTRINSIC_FOR_EACH = "mogg.for_each";

/// MOGG Intrinsic for a singleton for_each function that is used to drive
/// elemwise kernel execution
constexpr StringLiteral MOGG_INTRINSIC_ELEMWISE_FOR_EACH =
    "mogg.elemwise_for_each";

/// MOGG Instrinsic for the input / output lambda implementations.
constexpr StringLiteral MOGG_INTRINSIC_INPUT_FUSION_HOOK =
    "mogg.dps_input_fusion_hook";
constexpr StringLiteral MOGG_INTRINSIC_OUTPUT_FUSION_HOOK =
    "mogg.dps_output_fusion_hook";

/// MOGG Intrinsic for the ManagedTensorSlice _fused_load method.
constexpr StringLiteral MOGG_INTRINSIC_TENSOR_FUSED_LOAD =
    "mogg.tensor_fused_load";

/// MOGG Intrinsic for the ManagedTensorSlice _fused_store method.
constexpr StringLiteral MOGG_INTRINSIC_TENSOR_FUSED_STORE =
    "mogg.tensor_fused_store";

/// Track the pair of the decorator as it is seen in the LIT IR in its raw from
/// and the clean processed attribute which is added after it is processed.
struct MOGGDecorator {
  // The decorator to look for.
  StringLiteral decorator;

  // The attribute to replace it with.
  StringLiteral attr;
};

namespace Decorators {

// The decorators we will look for on the generator to identify it as a MO
// kernel.
constexpr StringLiteral REGISTER_INTERNAL_FUNCTION = "register_internal";

// Decorator to enforce ManagedTensorSlice specializing on the IO param,
// this allows us to transition to the new ManagedTensorSlice definition in a
// piecemeal fashion.
static constexpr StringLiteral ENFORCE_IO_PARAM = "enforce_io_param";

// Allow new attrs to be added without needing explicit decorator.
constexpr StringLiteral REGISTER_MOGG_INTRINSIC = "__mogg_intrinsic_attr";

} // namespace Decorators

//===----------------------------------------------------------------------===//
// DPS Tensor API definitions
//===----------------------------------------------------------------------===//

// Supported static methods under the registered structs.
static constexpr StringLiteral kMOGGKernelStructName = "mogg.kernel.struct";
static constexpr StringLiteral kMOGGExecuteFunctionLabel = "mogg.execute";
static constexpr StringLiteral kMOGGShapeFunctionLabel = "mogg.shape";
static constexpr StringLiteral kMOGGUpdateViewFunctionLabel =
    "mogg.update_view";
static constexpr StringLiteral kMOGGPyTorchFallbackFunctionLabel =
    "mogg.pytorch_fallback";

// An array of attributes, each of which correspond to an input argument type.
//
// If the associated argument is a parameterized tensor type, the attribute will
// be a dictionary attribute mapping the name of the unbound parameter
// (e.g. dtype) to the associated KGEN parameter decl ref.
//
// If the associated argument is not a supported tensor type, it will be the
// unit attribute.
static constexpr StringLiteral kKernelValueParameterAttrName =
    "mogg.value_params";

// An array of StringAttr corresponding to each input argument type.
//
// If the associated string is non-empty, it refers to the name of the
// tensor spec parameter. This is often added by a specialized Mojo pass
// like autoparameterization.
static constexpr StringLiteral kKernelTensorSpecParameterAttrName =
    "mogg.tensor_spec_params";

// The generator level label for operations which take in the special
// synchronous parameter. This is a hint that the runtime is running in
// synchronous mode. Often times this means the work done is trivial and
// the kernel may want to consider single-threading.
static constexpr StringLiteral kMOGGSynchronousLabel = "mogg.synchronous";
static constexpr StringLiteral kMOGGSynchronousParameterName = "_synchronous";

// The generator level label for the target device for the kernel, e.g. cpu.
static constexpr StringLiteral kMOGGTargetLabel = "mogg.target";
static constexpr StringLiteral kMOGGTargetParameterName = "target";

// The name of the output_rank parameter which we support lowering specially.
static constexpr StringLiteral kMOGGOutputRankParameterName = "output_rank";

// Generator level annotations for fusion.
static constexpr StringLiteral kMOGGViewKernel = "mogg.view_kernel";

// The number of DPS (Destination Passing Style) output operands there are.
static constexpr StringLiteral kMOGGNumDPSOutputs = "mogg.num_dps_outputs";

// Annotations which indicate whether a fusion lambda is present. Needed for
// accumulation-type ops like matmul and conv where only the last store should
// be to the lambda.
static constexpr StringLiteral kMOGGLambdasHaveFusionLabel =
    "mogg.lambdas_have_fusion";
static constexpr StringLiteral kMOGGLambdasHaveFusionParameterName =
    "lambdas_have_fusion";

// Trace name for the fused kernel.
// Need to be used in the kernel code like this:
// `with Trace[TraceLevel.OP, target=target](_trace_name):`
static constexpr StringLiteral kMOGGTraceNameLabel = "mogg.trace_name";
static constexpr StringLiteral kMOGGTraceNameParameterName = "_trace_name";

// An ArrayAttr of indices which correspond which operands have fusion enabled.
static constexpr StringLiteral kMOGGFusableArgs = "mogg.fusable_args";

// An ArrayAttr of of the encoded IOSpec value for all kernel arguments.
static constexpr StringLiteral kMOGGArgsIOSpecs = "mogg.args_io_specs";

static constexpr StringLiteral kMOGGBufferArgs = "mogg.buffer_args";

// Fusion interface implementation details.
static constexpr StringLiteral kMOGGInputLambdas = "_in_lambdas";
static constexpr StringLiteral kMOGGOutputLambdas = "_out_lambdas";
static constexpr StringLiteral kMOGGElementwiseLambda = "_elementwise_lambda";

// Mark a function as the outlined body of an elementwise kernel.
// During InlineLambdas those functions can be CSE.
static constexpr StringLiteral OUTLINED_ELEMW_ATTR = "mogg.outlined_elemw";

inline bool isExecuteFunc(Operation *gen) {
  return gen != nullptr && gen->hasAttr(kMOGGExecuteFunctionLabel);
}

inline bool isShapeFunc(Operation *gen) {
  return gen != nullptr && gen->hasAttr(kMOGGShapeFunctionLabel);
}

inline bool isUpdateViewFunc(Operation *gen) {
  return gen != nullptr && gen->hasAttr(kMOGGUpdateViewFunctionLabel);
}

inline bool isElemwiseForeachFunc(Operation *gen) {
  return gen != nullptr && gen->hasAttr(MOGG_INTRINSIC_ELEMWISE_FOR_EACH);
}

inline bool isExtensibilityFunc(Operation *gen) {
  return gen != nullptr && (gen->hasAttr(kMOGGExecuteFunctionLabel) ||
                            gen->hasAttr(kMOGGShapeFunctionLabel) ||
                            gen->hasAttr(kMOGGUpdateViewFunctionLabel) ||
                            gen->hasAttr(kMOGGPyTorchFallbackFunctionLabel));
}

//===----------------------------------------------------------------------===//
// DPS Tensor API type strings
//===----------------------------------------------------------------------===//

// The stored mojo type symbol name of Tensor type in extensibility kernels.
constexpr StringLiteral MOJO_DPS_TENSOR_TYPE_NAME =
    "tensor_internal::ManagedTensorSlice";

constexpr StringLiteral MOJO_INTERNAL_DPS_SIMD_TYPE_NAME = "stdlib::SIMD";

constexpr StringLiteral MOJO_INTERNAL_DPS_INT_TYPE_NAME = "stdlib::Int";

constexpr StringLiteral MOJO_INTERNAL_DPS_UINT_TYPE_NAME = "stdlib::UInt";

constexpr StringLiteral MOJO_INTERNAL_DPS_BOOL_TYPE_NAME = "stdlib::Bool";

// We support tuples of DPS tensors for operations with variadic input/outputs
constexpr StringLiteral MOJO_VARIADIC_TENSORS_NAME =
    "tensor_internal::VariadicTensors";

// We support lists of DPS tensors for a few operations
constexpr StringLiteral MOJO_TENSOR_LIST_NAME = "stdlib::List";

// The stored mojo type symbol name of device contexts in extensibility kernels.
constexpr StringLiteral MOJO_EXTENSIBILITY_API_DEVICE_CONTEXT_PTR_TYPE_NAME =
    "runtime::DeviceContextPtr";

// The stored mojo type symbol name of device contexts list in extensibility
// kernels.
constexpr StringLiteral
    MOJO_EXTENSIBILITY_API_DEVICE_CONTEXT_PTR_LIST_TYPE_NAME =
        "runtime::DeviceContextPtrList";

constexpr StringLiteral MOJO_INTERNAL_STATIC_STRING_TYPE_NAME =
    "stdlib::StringSlice";

//===----------------------------------------------------------------------===//
// Parameter Inference
//===----------------------------------------------------------------------===//

// Names of common params. We use these strings as the key to a parameter ref.
static constexpr llvm::StringLiteral kParameterMut = "mut";
static constexpr llvm::StringLiteral kParameterInput = "input";
static constexpr llvm::StringLiteral kParameterDType = "type";
static constexpr llvm::StringLiteral kParameterRank = "rank";
static constexpr llvm::StringLiteral kParameterIOSpec = "io_spec";
static constexpr llvm::StringLiteral kParameterStaticSpec = "static_spec";

// Used in variadics/tuple/SIMD types.
static constexpr llvm::StringLiteral kParameterSize = "size";
static constexpr llvm::StringLiteral kParameterStaticSpecs = "static_specs";

static constexpr bool kIOSpecImmutable = false;
static constexpr bool kIOSpecMutable = true;

static constexpr int64_t kIOSpecIOOutput = 0;
static constexpr int64_t kIOSpecIOInput = 1;
static constexpr int64_t kIOSpecIOFusedInput = 2;
static constexpr int64_t kIOSpecIOFusedOutput = 3;
static constexpr int64_t kIOSpecIOFusedComputeOutput = 31;

static constexpr int64_t kIOSpecIOUnknown = -1;

static constexpr llvm::StringLiteral kInputTensor = "InputTensor";
static constexpr llvm::StringLiteral kOutputTensor = "OutputTensor";
static constexpr llvm::StringLiteral kMutableInputTensor = "MutableInputTensor";
static constexpr llvm::StringLiteral kFusedInputTensor = "FusedInputTensor";
static constexpr llvm::StringLiteral kFusedOutputTensor = "FusedOutputTensor";
static constexpr llvm::StringLiteral kFusedComputeOutputTensor =
    "FusedComputeOutputTensor";

enum class IOSpec {
  InputTensor,        // Input tensor, read-only
  OutputTensor,       // Output tensor, write-only
  MutableInputTensor, // Input tensor that can be modified
  FusedInputTensor,
  FusedOutputTensor,
  FusedComputeOutputTensor,
};

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_MOGGDECORATORS_H
