//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TRANSFORMUTILS_MANGLINGUTILS_H
#define KGEN_TRANSFORMUTILS_MANGLINGUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN {
class GeneratorOpInterface;

/// Returns a simplified serialization of a parameter that is more readable.
/// Eventually this should be used by `mangleParameterValues` (with MOCO-945),
/// but today it does not guarantee unique serialization for type-values that
/// are identical except for the vtable.
void prettyPrintParameter(TypedAttr value, raw_ostream &os);

/// This returns a name to use when the specified generator is specialized
/// with the specified input parameters.
std::string
mangleParameterValues(GeneratorOpInterface generator,
                      ArrayRef<TypedAttr> inputParamValues,
                      function_ref<std::string(StringRef)> getPrefix);
} // namespace M::KGEN

#endif // KGEN_TRANSFORMUTILS_MANGLINGUTILS_H
