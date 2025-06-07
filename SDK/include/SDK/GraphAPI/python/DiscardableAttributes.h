//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_GRAPHAPI_PYTHON_DISCARDABLEATTRIBUTES_H
#define SDK_GRAPHAPI_PYTHON_DISCARDABLEATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "nanobind/nanobind.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

namespace M::Graph::Python {

struct DiscardableAttributes {
  mlir::Operation *op;
  DiscardableAttributes(mlir::Operation *op) : op(op) {}
};

} // namespace M::Graph::Python

#endif // SDK_GRAPHAPI_PYTHON_DISCARDABLEATTRIBUTES_H
