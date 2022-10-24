//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SEARCH_IMPL_H
#define SEARCH_IMPL_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/ForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"

namespace M::KGEN {
/// Search for the fastest specialization of the given interface. On success,
/// returns an index into the `specializations` vector - that is the fastest
/// implementation.
ErrorOr<size_t> selectFastestFunction(GeneratorInterfaceOp itf,
                                      SymbolTable &symtab,
                                      ArrayRef<FuncOp> specializations);

/// Given a list of valid specializations for an interface, select the best
/// specialization according to a user-defined evaluator function.
ErrorOr<size_t> evaluateSpecializations(FuncOp evaluator, SymbolTable &symtab,
                                        ArrayRef<FuncOp> specializations);
} // namespace M::KGEN

#endif // SEARCH_IMPL_H
