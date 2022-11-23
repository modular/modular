//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SEARCH_IMPL_H
#define SEARCH_IMPL_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/ForwardDecls.h"

namespace M::LLCL {
class Runtime;
}

namespace M::KGEN {
/// Given a list of valid specializations for an interface, select the best
/// specialization according to a user-defined evaluator function.
ErrorOr<size_t> evaluateSpecializations(FuncOp evaluator, SymbolTable &symtab,
                                        LLCL::Runtime &runtime,
                                        ArrayRef<FuncOp> specializations);
} // namespace M::KGEN

#endif // SEARCH_IMPL_H
