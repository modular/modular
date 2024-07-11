//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoLanguageRuntime.h"
#include "../Utils/Errors.h"
#include "Formatters/MojoDecoratorBasedTypeFormatter.h"
#include "Formatters/MojoKGENVariantTypeFormatter.h"
#include "Formatters/MojoListTypeFormatter.h"
#include "lldb/API/SBValue.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/VectorType.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace M::KGEN::Mojo;

LLDB_PLUGIN_DEFINE(MojoLanguageRuntime)

char MojoLanguageRuntime::ID = 0;

void MojoLanguageRuntime::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Mojo Language Runtime",
                                CreateInstance);
}

void MojoLanguageRuntime::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::LanguageRuntime *
MojoLanguageRuntime::CreateInstance(lldb_private::Process *process,
                                    lldb::LanguageType language) {
  if (language == lldb::eLanguageTypeMojo)
    return new MojoLanguageRuntime(process);
  return nullptr;
}

BreakpointResolverSP
MojoLanguageRuntime::CreateExceptionResolver(const BreakpointSP &bkpt,
                                             bool catch_bp, bool throw_bp) {
  std::vector<const char *> exceptionNames;
  exceptionNames.push_back("__mojo_debugger_raise_hook");
  BreakpointResolverSP resolver(new BreakpointResolverName(
      bkpt, exceptionNames.data(), exceptionNames.size(), eFunctionNameTypeFull,
      eLanguageTypeMojo, 0, eLazyBoolNo));
  BreakpointHitCallback callback =
      [](void *baton, StoppointCallbackContext *context,
         lldb::user_id_t break_id, lldb::user_id_t break_loc_id) -> bool {
    // Unlike C++, which drops you at `__cxa_throw` showing you some weird raw
    // instructions, we prefer to step out and drop you right at your `raise`
    // statement.
    context->exe_ctx_ref.GetThreadSP()->StepOut();
    return true;
  };
  lldb::BatonSP callback_baton_sp;
  resolver->GetBreakpoint()->SetCallback(callback, callback_baton_sp);

  return resolver;
}
