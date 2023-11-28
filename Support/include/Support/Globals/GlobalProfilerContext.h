//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/SymbolExport.h"

namespace M {
namespace ProfilingDetail {
struct GlobalProfilerContext;
}

namespace Globals {

extern MODULAR_CXX_EXPORT M::ProfilingDetail::GlobalProfilerContext *
getGlobalProfilerContext();

extern MODULAR_CXX_EXPORT M::ProfilingDetail::GlobalProfilerContext *
exchangeGlobalProfilerContext(M::ProfilingDetail::GlobalProfilerContext *ctx);

} // namespace Globals

} // namespace M
