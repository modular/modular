//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_GLOBALS_GLOBALS_H
#define SUPPORT_GLOBALS_GLOBALS_H

#include "Support/SymbolExport.h"

#include <functional>

namespace M {
namespace Detail {
class TypeInfoTable;
}

namespace ProfilingDetail {
struct GlobalProfilerContext;
}

namespace Globals {

extern MODULAR_CXX_EXPORT M::ProfilingDetail::GlobalProfilerContext *
getGlobalProfilerContext();

extern MODULAR_CXX_EXPORT M::ProfilingDetail::GlobalProfilerContext *
exchangeGlobalProfilerContext(M::ProfilingDetail::GlobalProfilerContext *ctx);

extern MODULAR_CXX_EXPORT Detail::TypeInfoTable &
getTypeInfoTableSingleton(const std::function<Detail::TypeInfoTable *()> &ctor);

} // namespace Globals

} // namespace M

#endif // SUPPORT_GLOBALS_GLOBALS_H
