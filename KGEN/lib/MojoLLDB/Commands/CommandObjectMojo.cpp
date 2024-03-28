//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectMojo.h"
#include "../REPL/MojoREPL.h"
#include "../ScriptingBridge/SBClassUtils.h"
#include "../TypeSystem/MojoTypeSystem.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Telemetry/Telemetry.h"
#include "lldb/Target/Target.h"

using namespace M;
using namespace M::KGEN::Mojo;
using namespace lldb;

namespace {
//===----------------------------------------------------------------------===//
// CommandREPLHelp: mojo help repl
//===----------------------------------------------------------------------===//
class CommandREPLHelp : public SBCommandPluginInterface {
public:
  bool DoExecute(SBDebugger debugger, char **command,
                 SBCommandReturnObject &result) override {
    result.AppendMessage(MojoREPL::GetHelpPrologue());
    return true;
  }
};

//===----------------------------------------------------------------------===//
// CommandDebugHelp: mojo help debug
//===----------------------------------------------------------------------===//
class CommandDebugHelp : public SBCommandPluginInterface {
public:
  bool DoExecute(SBDebugger debugger, char **command,
                 SBCommandReturnObject &result) override {
    result.AppendMessage(R"(
You can use LLDB to debug Mojo programs and its feature set is constantly
growing.

This is a non-comprehensive list of features currently supported:
- Source breakpoints
- Stack frame unwinding
- Variable printing
- Stepping

This is a non-comprehensive list of features not yet supported:
- Symbol breakpoints
- Capturing return values when stepping out of functions
- Expression evaluation
- Conditional breakpoints

Finally, we encourage you to submit feature requests and error reports in
https://github.com/modularml/mojo/issues.
)");
    result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
    return true;
  }
};

//===----------------------------------------------------------------------===//
// CommandStats: mojo stats
//
// Telemetry subcommand:
//   This subcommand logs the given event using Modular's telemetry.
//
//   mojo stats telemetry <event> <interface>
//     interface: vscode | cli
//
//===----------------------------------------------------------------------===//
class CommandStats : public SBCommandPluginInterface {
public:
  CommandStats(ContextRef ctx) : ctx(std::move(ctx)) {}

  bool DoExecute(SBDebugger debugger, char **command,
                 SBCommandReturnObject &result) override {
#ifdef MODULAR_ENABLE_TELEMETRY
    SmallVector<StringRef> args;
    for (char **it = command; it && *it; ++it)
      args.push_back(*it);

    // `telemetry` is not a proper subcommand to hide it from the help and
    // autocompletion results of LLDB.
    if (args.size() == 3 && args[0] == "telemetry") {
      StringRef event = args[1];
      StringRef interface = args[2];

      auto &telemetryCtx = *ctx->get<M::Telemetry::TelemetryContext>();
      auto logger = telemetryCtx.getLogger("debugger");
      logger->emitL1Event(event, {{"interface", interface}});
      result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
      return true;
    }
#endif // MODULAR_ENABLE_TELEMETRY
    result.SetStatus(lldb::eReturnStatusFailed);
    return false;
  }

  ContextRef ctx;
};

} // namespace

void M::KGEN::Mojo::registerMojoCommands(SBDebugger debugger, ContextRef ctx) {
  SBCommandInterpreter interpreter = debugger.GetCommandInterpreter();
  SBCommand root = interpreter.AddMultiwordCommand(
      "mojo", "Commands related to the Mojo language support.");

  root.AddCommand("statistics", new CommandStats(ctx),
                  "Commands related to statistics of Mojo");

  SBCommand help = root.AddMultiwordCommand(
      "help", "Display help information about various "
              "components of the Mojo support in LLDB.");
  help.AddCommand("repl", new CommandREPLHelp(), "mojo help repl");
  help.AddCommand("debug", new CommandDebugHelp(), "mojo help debug");
}
