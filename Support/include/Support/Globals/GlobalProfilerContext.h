//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/SymbolExport.h"

namespace M {
struct GlobalProfilerContext;

namespace Globals {

extern MODULAR_CXX_EXPORT M::GlobalProfilerContext *getGlobalProfilerContext();

extern MODULAR_CXX_EXPORT void
setGlobalProfilerContext(M::GlobalProfilerContext *ctx);

extern MODULAR_CXX_EXPORT M::GlobalProfilerContext *
exchangeGlobalProfilerContext(M::GlobalProfilerContext *ctx);

} // namespace Globals

} // namespace M
