//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains variuous utilities and defines for the Mojo LLDB plugin.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_PLUGIN_H
#define KGEN_LIB_MOJOLLDB_PLUGIN_H

#include "lldb/lldb-enumerations.h"

namespace M::KGEN::Mojo {
/// TODO(#11553): While the language is private we can't yet define a specific
/// language type. Until then, pretend that we're Go.
static inline constexpr lldb::LanguageType eLanguageTypeMojo =
    lldb::eLanguageTypeGo;
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_PLUGIN_H
