//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// DeflateSymbols
//===----------------------------------------------------------------------===//

def DeflateSymbols : Pass<"deflate-symbols", "mlir::ModuleOp"> {
  let summary = "Deflate symbols and put them in the cache.";
  let description = [{
    This pass deflates symbols and replaces them with `cache.symbol`
    ops, placing regions in the cache.
  }];

  let options = [
    Option<"cacheDir", "cache-dir", "std::string", "",
           "The cache directory to use.">,
  ];

  let dependentDialects = ["CacheDialect", "mlir::func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// InflateSymbols
//===----------------------------------------------------------------------===//

def InflateSymbols : Pass<"inflate-symbols", "mlir::ModuleOp"> {
  let summary = "Inflate symbols from the cache.";
  let description = [{
    This pass inflates symbols from the cache and replaces the original
    operation.
  }];

  let options = [
    Option<"cacheDir", "cache-dir", "std::string", "",
           "The cache directory to use.">,
  ];

  let dependentDialects = ["CacheDialect", "mlir::func::FuncDialect"];
}
