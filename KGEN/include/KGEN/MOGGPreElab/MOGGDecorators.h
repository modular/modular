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

inline bool isV1ShapeFunc(Operation *gen) {
  return gen != nullptr && gen->hasAttr(shapeFuncRegistrationAttr);
}

constexpr StringLiteral SLICED_ATTR = "mogg.sliced";
constexpr StringLiteral ALLOCS_ATTR = "mogg.allocs";
constexpr StringLiteral IS_VIEW_ATTR = "mogg.view";
constexpr StringLiteral OUTLINED_ATTR = "mogg.outlined";

/// Tracks the mojo parameter value for each of the input parameters.
constexpr StringLiteral MOGG_ARG_PARAMS = "mogg.arg_params";
constexpr StringLiteral MOGG_ARG_RESULT_PARAMS = "mogg.result_params";
constexpr StringLiteral MOGG_ARG_TYPE_NAMES = "mogg.arg_type_names";

// The names as they appear in the lit source.
constexpr StringLiteral MOGG_ARG_SRC_NAMES = "mogg.arg_src_names";

/// Tracks the mojo trait conformances of each argument and result type.
constexpr StringLiteral MOGG_ARGUMENT_CONFORMANCES = "mogg.arg_conformances";
constexpr StringLiteral MOGG_RESULT_CONFORMANCES = "mogg.result_conformances";

/// MOGG Intrinsic for the register kernel decorator.
constexpr StringLiteral MOGG_INTRINSIC_REGISTER = "mogg.intrinsic_register";

/// MOGG Intrinsic for the specsof function.
constexpr StringLiteral MOGG_INTRINSIC_TENSOR_SPEC_HOOK =
    "mogg.intrinsic_tensor_spec_hook";

/// MOGG Intrinsic for the elementwise kernel decorator.
constexpr StringLiteral MOGG_INTRINSIC_ELEMENTWISE = "mogg.elementwise";

/// MOGG Intrinsic decorator to indicates which I/Os can be fused.
constexpr StringLiteral MOGG_INTRINSIC_ENABLE_FUSION_FOR =
    "mogg.enable_fusion_for";

/// MOGG Intrinsic for the view kernel decorator.
constexpr StringLiteral MOGG_INTRINSIC_VIEW_KERNEL = "mogg.view_kernel";

/// MOGG Intrinsic for the for_each function
constexpr StringLiteral MOGG_INTRINSIC_FOR_EACH = "mogg.for_each";

/// MOGG Instrinsic for the input / output lambda implementations.
constexpr StringLiteral MOGG_INTRINSIC_INPUT_FUSION_HOOK =
    "mogg.dps_input_fusion_hook";
constexpr StringLiteral MOGG_INTRINSIC_OUTPUT_FUSION_HOOK =
    "mogg.dps_output_fusion_hook";

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
constexpr StringLiteral REGISTER_KERNEL = "mogg_register";
constexpr StringLiteral REGISTER_OVERRIDE = "mogg_register_override";
constexpr StringLiteral REGISTER_PUBLIC_OVERRIDE = "op";

constexpr StringLiteral REGISTER_SHAPE_FUNC = "mogg_register_shape_func";

// Allow new attrs to be added without needing explicit decorator.
constexpr StringLiteral REGISTER_MOGG_INTRINSIC = "__mogg_intrinsic_attr";

// MOGG API V1 hooks.
constexpr MOGGDecorator ELEMENTWISE{"mogg_elementwise", "mogg.elementwise"};
constexpr MOGGDecorator ELEMENTWISE_PUBLIC{"elementwise", "mogg.elementwise"};
constexpr MOGGDecorator VIEW{"mogg_view_op", IS_VIEW_ATTR};
constexpr MOGGDecorator TAKES_INDICES{"mogg_takes_indices",
                                      "mogg.takes_indices"};

// Tensor API hooks.

constexpr MOGGDecorator TENSOR_ALLOC{"mogg_tensor_allocator",
                                     "mogg.tensor_alloc"};
constexpr MOGGDecorator TENSOR_COPY{"mogg_tensor_copy_constructor",
                                    "mogg.tensor_copy_construct"};
constexpr MOGGDecorator TENSOR_DECONSTRUCT{"mogg_tensor_deconstructor",
                                           "mogg.tensor_destruct"};
constexpr MOGGDecorator ELEM_HOOK{"mogg_elementwise_hook", "mogg.elem_hook"};

constexpr MOGGDecorator ENABLE_FUSION{"mogg_enable_fusion",
                                      "mogg.enable_fusion"};
constexpr MOGGDecorator INPUT_FUSION{"mogg_input_fusion_hook",
                                     "mogg.input_fusion_hook"};
constexpr MOGGDecorator OUTPUT_FUSION{"mogg_output_fusion_hook",
                                      "mogg.output_fusion_hook"};

} // namespace Decorators

//===----------------------------------------------------------------------===//
// DPS Tensor API definitions
//===----------------------------------------------------------------------===//

static constexpr StringLiteral kMOGGExecuteFunctionLabel = "mogg.execute";
static constexpr StringLiteral kMOGGShapeFunctionLabel = "mogg.shape";
static constexpr StringLiteral kMOGGInitializeOutputFunctionLabel =
    "mogg.initialize_output";

static constexpr StringLiteral kKernelTensorParameterAttrName =
    "mogg.tensor_params";
static constexpr StringLiteral kKernelTensorSpecParameterAttrName =
    "mogg.tensor_spec_params";
static constexpr StringLiteral kMOGGSynchronousParameterName = "synchronous";
static constexpr StringLiteral kMOGGSynchronousLabel = "mogg.synchronous";
static constexpr StringLiteral kMOGGTargetParameterName = "target";
static constexpr StringLiteral kMOGGTargetLabel = "mogg.target";
static constexpr StringLiteral kMOGGElementFunction = "mogg.elementwise";
static constexpr StringLiteral kMOGGViewKernel = "mogg.view_kernel";

static constexpr StringLiteral kMOGGFusableArgs = "mogg.fusable_args";
static constexpr StringLiteral kMOGGInputLambdas = "_in_lambdas";
static constexpr StringLiteral kMOGGOutputLambdas = "_out_lambdas";
static constexpr StringLiteral kMOGGElementwiseLambda = "_elementwise_lambda";

inline bool isExecuteFunc(Operation *gen) {
  return gen != nullptr && gen->hasAttr(kMOGGExecuteFunctionLabel);
}

inline bool isShapeFunc(Operation *gen) {
  return gen != nullptr && gen->hasAttr(kMOGGShapeFunctionLabel);
}

inline bool isDPSKernel(Operation *gen) {
  return gen != nullptr && (gen->hasAttr(kMOGGExecuteFunctionLabel) ||
                            gen->hasAttr(kMOGGShapeFunctionLabel) ||
                            gen->hasAttr(kMOGGInitializeOutputFunctionLabel));
}

//===----------------------------------------------------------------------===//
// DPS Tensor API type strings
//===----------------------------------------------------------------------===//

// The stored mojo type symbol name of Tensor type in extensibility kernels.
constexpr StringLiteral MOJO_DPS_TENSOR_TYPE_NAME =
    "tensor_utils::ManagedTensorSlice";

constexpr StringLiteral MOJO_INTERNAL_DPS_TENSOR_TYPE_NAME =
    "tensor_utils_internal::ManagedTensorSlice";

//===----------------------------------------------------------------------===//
// MOGG Tensor API type strings
//===----------------------------------------------------------------------===//

constexpr StringLiteral MOJO_MOGG_TENSOR_TYPE_NAME = "MOGGTensor::Tensor";

// The stored mojo type symbol name of the mojo ctx in extensibility kernels.
constexpr StringLiteral MOJO_EXTENSIBILITY_API_CALL_CONTEXT_PTR_TYPE_NAME =
    "runtime::MojoCallContextPtr";

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_MOGGDECORATORS_H
