//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOGGPREELAB_MOGGDECORATORS_H
#define KGEN_LIB_MOGGPREELAB_MOGGDECORATORS_H

#include "KGEN/KGENDialect/KGENOps.h"

namespace M::KGEN::MOGGPreElab {

// The decorators we will look for on the generator to identify it as a MO
// kernel.
constexpr StringLiteral registerDecorator = "register::register::mogg_register";
// TODO(#27757): Temporary as transition to Mojo async/await.
constexpr StringLiteral willBecomeAsyncDecorator =
    "register::register::mogg_will_become_async";
constexpr StringLiteral registerOverrideDecorator =
    "register::register::mogg_register_override";

constexpr StringLiteral tensorAllocDecorator =
    "register::register::mogg_tensor_allocator";
constexpr StringLiteral tensorCopyConstructDecorator =
    "register::register::mogg_tensor_copy_constructor";
constexpr StringLiteral tensorDeconstructDecorator =
    "register::register::mogg_tensor_deconstructor";

constexpr StringLiteral elementwiseHook =
    "register::register::mogg_elementwise_hook";
constexpr StringLiteral tensorEnableFusion =
    "register::register::mogg_enable_fusion";
constexpr StringLiteral tensorInputFusionHook =
    "register::register::mogg_input_fusion_hook";
constexpr StringLiteral tensorOutputFusionHook =
    "register::register::mogg_output_fusion_hook";

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_MOGGDECORATORS_H
