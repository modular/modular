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
constexpr StringLiteral registerDecorator =
    "stdlib::utils::_annotations::mogg_register";
// TODO(#27757): Temporary as transition to Mojo async/await.
constexpr StringLiteral willBecomeAsyncDecorator =
    "stdlib::utils::_annotations::mogg_will_become_async";
constexpr StringLiteral registerOverrideDecorator =
    "stdlib::utils::_annotations::mogg_register_override";

constexpr StringLiteral tensorAllocDecorator =
    "stdlib::utils::_annotations::mogg_tensor_allocator";
constexpr StringLiteral tensorCopyConstructDecorator =
    "stdlib::utils::_annotations::mogg_tensor_copy_constructor";
constexpr StringLiteral tensorDeconstructDecorator =
    "stdlib::utils::_annotations::mogg_tensor_deconstructor";

constexpr StringLiteral elementwiseHook =
    "stdlib::utils::_annotations::mogg_elementwise_hook";
constexpr StringLiteral tensorEnableFusion =
    "stdlib::utils::_annotations::mogg_enable_fusion";
constexpr StringLiteral tensorInputFusionHook =
    "stdlib::utils::_annotations::mogg_input_fusion_hook";
constexpr StringLiteral tensorOutputFusionHook =
    "stdlib::utils::_annotations::mogg_output_fusion_hook";

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_MOGGDECORATORS_H
