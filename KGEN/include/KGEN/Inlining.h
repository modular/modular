//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_INLINING_H
#define KGEN_INLINING_H

#include "KGEN/KGENDialect/KGENParameters.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN {
class CallOp;
class GeneratorOp;

/// Given a call (to a GeneratorOp) and a ParameterUseDefGraph that describes
/// the parameter graph of the call's parent DeclInterface, inline the callee
/// into the call's context. This will mangle input parameters as necessary to
/// ensure that there are no conflicts.
void inlineGeneratorCall(CallOp call, GeneratorOp callee, InlineLevel level,
                         ParameterUseDefGraph &topLevelGraph,
                         const ParameterUseDefGraph &calleeParams,
                         const llvm::SetVector<StringAttr> &calleeDecls,
                         DenseSet<const void *> &manglerCache);

} // namespace M::KGEN

#endif // KGEN_INLINING_H
