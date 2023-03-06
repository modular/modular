//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_INLINING_H
#define KGEN_INLINING_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "Support/ADT/SmartVariant.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN {
/// Given a call (to a GeneratorOp) and a ParameterUseDefGraph that describes
/// the parameter graph of the call's parent DeclInterface, inline the callee
/// into the call's context. This will mangle input parameters as necessary to
/// ensure that there are no conflicts.
LogicalResult inlineGeneratorCall(
    KGENCallOpInterface topCall, ParameterUseDefGraph &topLevelGraph,
    ParameterCollector::Analysis &paramCache,
    function_ref<ParameterUseDefGraph &(ParameterCollector::Analysis &,
                                        GeneratorOp)>
        getGraph,
    function_ref<GeneratorOp(KGENCallOpInterface)> lookupCallee);
} // namespace M::KGEN

#endif // KGEN_INLINING_H
