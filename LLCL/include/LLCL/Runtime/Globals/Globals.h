//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_GLOBALS_H
#define LLCL_RUNTIME_GLOBALS_H

#include "Support/SymbolExport.h"

#include <functional>

namespace M::LLCL {
class Runtime;

namespace Detail {
class TypeInfoTable;
class RuntimeTable;
} // namespace Detail

} // namespace M::LLCL

namespace M::LLCL::Globals {

extern MODULAR_CXX_EXPORT Detail::TypeInfoTable &
getTypeInfoTableSingleton(const std::function<Detail::TypeInfoTable *()> &ctor);

extern MODULAR_CXX_EXPORT Detail::RuntimeTable &
getRuntimeTableSingleton(const std::function<Detail::RuntimeTable *()> &ctor);

} // namespace M::LLCL::Globals

#endif
