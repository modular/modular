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
                                      ModuleOp primaryModule,
                                      ArrayRef<FuncOp> specializations);
} // namespace M::KGEN

#endif // SEARCH_IMPL_H
