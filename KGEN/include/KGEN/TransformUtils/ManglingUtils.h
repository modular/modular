//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TRANSFORMUTILS_MANGLINGUTILS_H
#define KGEN_TRANSFORMUTILS_MANGLINGUTILS_H

#include "KGEN/KGENDialect/KGENOps.h"

namespace M::KGEN {

/// This returns a name to use when the specified generator is specialized
/// with the specified input parameters.
std::string mangleParameterValues(KGEN::GeneratorOp generator,
                                  ArrayRef<TypedAttr> inputParamValues);

} // namespace M::KGEN

#endif // KGEN_TRANSFORMUTILS_MANGLINGUTILS_H
