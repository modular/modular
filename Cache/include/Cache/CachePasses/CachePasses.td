//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// InflateSymbols
//===----------------------------------------------------------------------===//

def InflateSymbols : Pass<"inflate-symbols", "mlir::ModuleOp"> {
  let summary = "Inflate symbols from the cache.";
  let description = [{
    This pass inflates symbols' regions from the cache and restores the
    original operation.
  }];

  let options = [
    Option<"cacheDir", "cache-dir", "std::string", "",
           "The cache directory to use.">,
  ];

  let dependentDialects = ["CacheDialect", "mlir::func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// InflateConstants
//===----------------------------------------------------------------------===//

def InflateConstants : Pass<"inflate-constants", "mlir::ModuleOp"> {
  let summary = "Inflate constants from the cache.";
  let description = [{
    This pass inflates constants by replacing the MLIR resource with a
    reference to the cached object.
  }];

  let options = [
    Option<"cacheDir", "cache-dir", "std::string", "",
           "The cache directory to use.">,
  ];

  let dependentDialects = ["CacheDialect", "mlir::arith::ArithDialect"];
}
