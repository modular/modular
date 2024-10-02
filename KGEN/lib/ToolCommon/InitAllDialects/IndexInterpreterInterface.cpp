//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/InitAllDialects/IndexInterpreterInterface.h"

using namespace M;
using namespace KGEN;

/// Compare two integers according to the comparison predicate.
static bool compareIndices(const APInt &lhs, const APInt &rhs,
                           mlir::index::IndexCmpPredicate pred) {
  switch (pred) {
  case mlir::index::IndexCmpPredicate::EQ:
    return lhs.eq(rhs);
  case mlir::index::IndexCmpPredicate::NE:
    return lhs.ne(rhs);
  case mlir::index::IndexCmpPredicate::SGE:
    return lhs.sge(rhs);
  case mlir::index::IndexCmpPredicate::SGT:
    return lhs.sgt(rhs);
  case mlir::index::IndexCmpPredicate::SLE:
    return lhs.sle(rhs);
  case mlir::index::IndexCmpPredicate::SLT:
    return lhs.slt(rhs);
  case mlir::index::IndexCmpPredicate::UGE:
    return lhs.uge(rhs);
  case mlir::index::IndexCmpPredicate::UGT:
    return lhs.ugt(rhs);
  case mlir::index::IndexCmpPredicate::ULE:
    return lhs.ule(rhs);
  case mlir::index::IndexCmpPredicate::ULT:
    return lhs.ult(rhs);
  }
  llvm_unreachable("unhandled IndexCmpPredicate predicate");
}

ErrorTreeOrSuccess
CmpOpInterpretInterface::interpret(mlir::index::CmpOp cmpOp,
                                   ArrayRef<Attribute> operands,
                                   InterpreterState &state) {
  assert(operands.size() == 2 && "cmp expected two operands");
  IntegerAttr lhs = dyn_cast_if_present<mlir::IntegerAttr>(operands[0]);
  IntegerAttr rhs = dyn_cast_if_present<mlir::IntegerAttr>(operands[1]);
  uint64_t targetBitwidth = state.getTarget().resolveIndexBitWidth();
  auto result =
      BoolAttr::get(cmpOp.getContext(),
                    compareIndices(lhs.getValue().truncSSat(targetBitwidth),
                                   rhs.getValue().truncSSat(targetBitwidth),
                                   cmpOp.getPred()));
  state.mapResults(result);
  return success();
}
