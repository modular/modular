//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_NANOBIND_DISCARDABLEATTRIBUTES_H
#define SUPPORT_NANOBIND_DISCARDABLEATTRIBUTES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

namespace M::Graph::Python {

struct DiscardableAttributes {
  mlir::Operation *op;
  DiscardableAttributes(mlir::Operation *op) : op(op) {}
};

} // namespace M::Graph::Python

#endif // SUPPORT_NANOBIND_DISCARDABLEATTRIBUTES_H
