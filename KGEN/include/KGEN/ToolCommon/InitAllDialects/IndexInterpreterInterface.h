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

struct CmpOpInterpretInterface
    : public IndexOpInterpretInterface<CmpOpInterpretInterface,
                                       mlir::index::CmpOp> {
  static ErrorTreeOrSuccess interpret(mlir::index::CmpOp op,
                                      ArrayRef<Attribute> operands,
                                      InterpreterState &state);
};

struct SubOpInterpretInterface
    : public IndexOpInterpretInterface<SubOpInterpretInterface,
                                       mlir::index::SubOp> {
  static ErrorTreeOrSuccess interpret(mlir::index::SubOp op,
                                      ArrayRef<Attribute> operands,
                                      InterpreterState &state);
};

} // namespace M::KGEN

#endif // KGEN_TOOLCOMMON_INITALLDIALECTS_INDEXINTERPRETERINTERFACE_H
