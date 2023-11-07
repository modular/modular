//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// SliceMOGGFuncs
//===----------------------------------------------------------------------===//

def SliceMOGGFuncs : Pass<"slice-mogg-funcs", "mlir::ModuleOp"> {
  let summary = "Slices shape and compute from mogg kernels";
  let description = [{
    A mogg domain specific pass which performs KGEN to KGEN transforms that
    the graph compiler can understand. The KGEN will be legal regardless.
    
    This description to expand as the functionality is added to the pass.
  }];
}
