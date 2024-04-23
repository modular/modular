//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOGGPREELAB_MOGGDECORATORS_H
#define KGEN_LIB_MOGGPREELAB_MOGGDECORATORS_H

#include "KGEN/KGENDialect/KGENUtils.h"
#include "llvm/ADT/StringRef.h"

namespace M::KGEN::MOGGPreElab {

// Attribute on generator ops to look for which marks the function as being a
// kernel.
constexpr llvm::StringLiteral kernelRegistrationAttr = "_mogg_kernel";

inline bool isKernel(GeneratorOp gen) {
  return gen != nullptr && gen->hasAttr(kernelRegistrationAttr);
}

constexpr llvm::StringLiteral shapeFuncRegistrationAttr = "_mogg_v1_shape_func";

inline bool isV1ShapeFunc(GeneratorOp gen) {
  return gen != nullptr && gen->hasAttr(shapeFuncRegistrationAttr);
}

// The decorators we will look for on the generator to identify it as a MO
// kernel.
constexpr llvm::StringLiteral DECORATOR_REGISTER_KERNEL =
    "register::register::mogg_register";
constexpr llvm::StringLiteral DECORATOR_REGISTER_OVERRIDE =
    "register::register::mogg_register_override";
constexpr llvm::StringLiteral DECORATOR_REGISTER_PUBLIC_OVERRIDE =
    "max::register::register::op";

// MOGG V1 shape function reg.
constexpr llvm::StringLiteral DECORATOR_REGISTER_SHAPE_FUNC =
    "register::register::mogg_register_shape_func";

// MOGG API V1 hooks.
constexpr llvm::StringLiteral DECORATOR_ELEMENTWISE =
    "register::register::mogg_elementwise";
constexpr llvm::StringLiteral DECORATOR_ELEMENTWISE_PUBLIC =
    "max::register::register::elementwise";
constexpr llvm::StringLiteral DECORATOR_VIEW =
    "register::register::mogg_view_op";
constexpr llvm::StringLiteral DECORATOR_TAKES_INDICES =
    "register::register::mogg_takes_indices";

// Tensor API hooks.
constexpr llvm::StringLiteral DECORATOR_TENSOR_ALLOC =
    "register::register::mogg_tensor_allocator";
constexpr llvm::StringLiteral DECORATOR_TENSOR_COPY_CONSTRUCT =
    "register::register::mogg_tensor_copy_constructor";
constexpr llvm::StringLiteral DECORATOR_TENSOR_DECONSTRUCT =
    "register::register::mogg_tensor_deconstructor";
constexpr llvm::StringLiteral DECORATOR_ELEM_HOOK =
    "register::register::mogg_elementwise_hook";
constexpr llvm::StringLiteral DECORATOR_ENABLE_FUSION_HOOK =
    "register::register::mogg_enable_fusion";
constexpr llvm::StringLiteral DECORATOR_INPUT_FUSION_HOOK =
    "register::register::mogg_input_fusion_hook";
constexpr llvm::StringLiteral DECORATOR_OUTPUT_FUSION_HOOK =
    "register::register::mogg_output_fusion_hook";

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_MOGGDECORATORS_H
