//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOGGPREELAB_PASSES_H
#define KGEN_MOGGPREELAB_PASSES_H

#include "Support/LLVMForwardDecls.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

//===----------------------------------------------------------------------===//
// MOGG Specific generated Pass Classes and Registration
//===----------------------------------------------------------------------===//
namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_MOGGPREELAB_PASSES_H
