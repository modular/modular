//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_IREVALUATORCONTEXT_H
#define KGEN_ELABORATOR_IREVALUATORCONTEXT_H

#include "KGEN/Interpreter/BytecodeInterpreter.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/Threading/Shared.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// IREvaluatorContext
//===----------------------------------------------------------------------===//

class IREvaluatorContext {
public:
  IREvaluatorContext(EnvAttr env, MLIRContext *mlirCtx,
                     InterpreterState *state);

protected:
  /// Evaluate an apply-like operator.
  FailureOr<TypedAttr> evaluateGetEnv(ParamOperatorAttr op);

  /// Evaluate POC::DataToStr "data_to_str" operator.
  FailureOr<TypedAttr> evaluateDataToStr(ParamOperatorAttr op);

  FailureOr<StringAttr> evaluateStringPart(TypedAttr part);

  /// The function to use to emit an error.
  std::function<void(ErrorTree)> emitError;

  /// The contextual location of an error.
  std::optional<Location> errorLoc;

  EnvAttr env;

  InterpreterState *state = nullptr;

private:
  MLIRContext *mlirCtx = nullptr;
};

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_IREVALUATORCONTEXT_H
