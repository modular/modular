//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file registers all the dialects in the KGEN library.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_INITALLDIALECTS_INDEXINTERPRETERINTERFACE_H
#define KGEN_TOOLCOMMON_INITALLDIALECTS_INDEXINTERPRETERINTERFACE_H

#include "KGEN/Interpreter/InterpreterDialect.h"
#include "KGEN/Interpreter/InterpreterInterface.h"
#include "KGEN/Interpreter/InterpreterState.h"
#include "KGEN/Interpreter/Utils.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"

namespace M::KGEN {

template <typename Interface, typename Concrete>
struct IndexOpInterpretInterface
    : public BytecodeInterpreterOpInterface::ExternalModel<
          IndexOpInterpretInterface<Interface, Concrete>, Concrete> {
  static OpBytecodeGenerator getBytecodeGenerator() {
    return {
        0, nullptr,
        +[](Operation *op, ArrayRef<Attribute> operands, const void *payload,
            InterpreterState &state) -> ErrorTreeOrSuccess {
          if (!state.getTarget()) {
            SmallVector<OpFoldResult> foldResults;
            if (LLVM_UNLIKELY(failed(op->fold(operands, foldResults))))
              return reportFoldError(op, operands, "failed to fold operation ");

            SmallVector<Attribute> results =
                llvm::map_to_vector(foldResults, [](OpFoldResult foldResult) {
                  return cast<Attribute>(foldResult);
                });
            state.mapResults(results);
            return success();
          }
          Concrete concrete = cast<Concrete>(op);
          return Interface::interpret(concrete, operands, state);
        }};
  }
};

//===----------------------------------------------------------------------===//
// IndexOpInterpretInterface Implementations
//===----------------------------------------------------------------------===//

template <typename IndexOpT>
struct IndexOpInterpretInterfaceImplementation
    : public IndexOpInterpretInterface<
          IndexOpInterpretInterfaceImplementation<IndexOpT>, IndexOpT> {
  static ErrorTreeOrSuccess interpret(IndexOpT op, ArrayRef<Attribute> operands,
                                      InterpreterState &state);
};

using CmpOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::CmpOp>;

using SubOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::SubOp>;

using ShlOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::ShlOp>;

using ShrSOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::ShrSOp>;

using ShrUOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::ShrUOp>;

using AndOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::AndOp>;

using CeilDivUOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::CeilDivUOp>;

using CeilDivSOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::CeilDivSOp>;

using DivUOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::DivUOp>;

using DivSOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::DivSOp>;

using MaxUOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::MaxUOp>;

using MaxSOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::MaxSOp>;

using MinUOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::MinUOp>;

using MinSOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::MinSOp>;

using MulOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::MulOp>;

using OrOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::OrOp>;

using RemSOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::RemSOp>;

using RemUOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::RemUOp>;

using XOrOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::XOrOp>;
} // namespace M::KGEN

#endif // KGEN_TOOLCOMMON_INITALLDIALECTS_INDEXINTERPRETERINTERFACE_H
