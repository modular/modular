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
/// and then append characters to the end of the symbol to keep it unique, if
/// name length is longer than charToKeep it truncates the sanitized name and
/// append _hash_hex(name) to the end making it at most 64 character long.
StringAttr sanitizeSymbolToAlnum(StringAttr name, size_t charToKeep = 32);

/// Like sanitizeSymbolToAlnum but replaces every run of invalid characters with
/// a single '_' without appending their encoded forms. This produces cleaner
/// PTX names when the source string uses separator characters (e.g. dots) that
/// are meaningful to humans but irrelevant after sanitisation. Long-name
/// hashing and digit-start fixup behave identically to sanitizeSymbolToAlnum.
StringAttr sanitizeSymbolToUnderscores(StringAttr name, size_t charToKeep = 32);

} // namespace M::KGEN

#endif // KGEN_NAMEMANGLING_H
