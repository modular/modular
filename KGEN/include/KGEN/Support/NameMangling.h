//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_NAMEMANGLING_H
#define KGEN_NAMEMANGLING_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// Name Mangling
//===----------------------------------------------------------------------===//

/// Many backends don't support arbitrary symbol names. This function will
/// losslessly re-mangle a symbol to only alnum characters and underscores.
/// The mangling scheme will replace all unsupported characters with underscores
/// and then append characters to the end of the symbol to keep it unique.
StringAttr sanitizeSymbolToAlnum(StringAttr name);

} // namespace M::KGEN

#endif // KGEN_NAMEMANGLING_H
