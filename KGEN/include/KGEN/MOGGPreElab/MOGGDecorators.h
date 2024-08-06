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

/// Tracks the mojo parameter value for each of the input parameters.
constexpr StringLiteral MOGG_ARG_PARAMS = "mogg.arg_params";
constexpr StringLiteral MOGG_ARG_RESULT_PARAMS = "mogg.result_params";
constexpr StringLiteral MOGG_ARG_TYPE_NAMES = "mogg.arg_type_names";

// The names as they appear in the lit source.
constexpr StringLiteral MOGG_ARG_SRC_NAMES = "mogg.arg_src_names";

constexpr StringLiteral REGISTER_TENSOR_SPEC_HOOK = "mogg.tensor_spec_hook";
/// Tracks the mojo trait conformances of each argument and result type.
constexpr StringLiteral MOGG_ARGUMENT_CONFORMANCES = "mogg.arg_conformances";
constexpr StringLiteral MOGG_RESULT_CONFORMANCES = "mogg.result_conformances";

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
constexpr StringLiteral REGISTER_MOGG_INTRINSIC = "mogg_intrinsic_attr";

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
static constexpr StringLiteral kKernelTensorParameterAttrName =
    "mogg.tensor_params";
static constexpr StringLiteral kMOGGSynchronousParameterName = "synchronous";
static constexpr StringLiteral kMOGGSynchronousLabel = "mogg.synchronous";
static constexpr StringLiteral kMOGGTargetParameterName = "target";
static constexpr StringLiteral kMOGGTargetLabel = "mogg.target";

inline bool isExecuteFunc(Operation *gen) {
  return gen != nullptr && gen->hasAttr(kMOGGExecuteFunctionLabel);
}

inline bool isShapeFunc(Operation *gen) {
  return gen != nullptr && gen->hasAttr(kMOGGShapeFunctionLabel);
}

inline bool isDPSKernel(Operation *gen) {
  return gen != nullptr && (gen->hasAttr(kMOGGExecuteFunctionLabel) ||
                            gen->hasAttr(kMOGGShapeFunctionLabel));
}

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_MOGGDECORATORS_H
