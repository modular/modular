//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOGGPREELAB_USERLIBRARYCHECKER_H
#define KGEN_LIB_MOGGPREELAB_USERLIBRARYCHECKER_H

#include "KGEN/KGENDialect/KGENOps.h"

namespace M::KGEN::MOGGPreElab {

struct CallGraph;
struct CallGraphNode;

/// Whether the generator operation has a decorator with the given annotation.
bool hasDecorator(GeneratorOp gen, StringLiteral annotation);

/// Whether the decorator's name is (starts with) the specific annotation.
bool isDecorator(TypedAttr decorator, StringLiteral annotation);

/// Whether the generator operation contains any decorator with any of the given
/// annotations.
bool hasAnyDecorator(GeneratorOp gen,
                     llvm::ArrayRef<StringLiteral> annotations);

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_USERLIBRARYCHECKER_H
