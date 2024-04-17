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

// The decorators we will look for on the generator to identify it as a MO
// kernel.
constexpr llvm::StringLiteral registerDecorator =
    "register::register::mogg_register";
// TODO(#27757): Temporary as transition to Mojo async/await.
constexpr llvm::StringLiteral willBecomeAsyncDecorator =
    "register::register::mogg_will_become_async";
constexpr llvm::StringLiteral registerOverrideDecorator =
    "register::register::mogg_register_override";

constexpr llvm::StringLiteral tensorAllocDecorator =
    "register::register::mogg_tensor_allocator";
constexpr llvm::StringLiteral tensorCopyConstructDecorator =
    "register::register::mogg_tensor_copy_constructor";
constexpr llvm::StringLiteral tensorDeconstructDecorator =
    "register::register::mogg_tensor_deconstructor";

constexpr llvm::StringLiteral elementwiseHook =
    "register::register::mogg_elementwise_hook";
constexpr llvm::StringLiteral tensorEnableFusion =
    "register::register::mogg_enable_fusion";
constexpr llvm::StringLiteral tensorInputFusionHook =
    "register::register::mogg_input_fusion_hook";
constexpr llvm::StringLiteral tensorOutputFusionHook =
    "register::register::mogg_output_fusion_hook";

inline bool hasRegisteredKernelDecorator(GeneratorOp gen) {
  return hasAnyDecorator(gen, {registerDecorator, registerOverrideDecorator});
}

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_MOGGDECORATORS_H
