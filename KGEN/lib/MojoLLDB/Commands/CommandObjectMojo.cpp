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
// CommandBreakOnRaise: mojo break-on-raise ([enable|disable])
//===----------------------------------------------------------------------===//
class CommandBreakOnRaise : public SBCommandPluginInterface {
public:
  bool DoExecute(SBDebugger debugger, char **command,
                 SBCommandReturnObject &result) override {
    SmallVector<StringRef> args;
    for (char **it = command; it && *it; ++it)
      args.push_back(*it);

    if (args.empty())
      args.push_back("enable");

    if (args.size() != 1) {
      result.SetError("invalid number of arguments");
      return false;
    }

    if (args[0] == "enable") {
      getOrCreateBreakpoint(debugger).SetEnabled(true);
      return true;
    } else if (args[0] == "disable") {
      getOrCreateBreakpoint(debugger).SetEnabled(false);
      return true;
    } else {
      result.SetError("invalid argument");
      return false;
    }
  }

private:
  SBBreakpoint getOrCreateBreakpoint(SBDebugger &debugger) {
    if (!bp)
      bp = debugger.GetSelectedTarget().BreakpointCreateForException(
          lldb::eLanguageTypeMojo, /*catch_bp=*/false, /*throw_bp=*/true);

    return bp;
  }

  SBBreakpoint bp;
};

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

  root.AddCommand("break-on-raise", new CommandBreakOnRaise(),
                  "Enables or disables breakpoints on raise statements. If no "
                  "arguments are specified, this feature will be enabled.",
                  "mojo break-on-raise ([enable|disable])");
  root.AddCommand("statistics", new CommandStats(ctx),
                  "Internal commands related to statistics of Mojo");

  SBCommand help = root.AddMultiwordCommand(
      "help", "Display help information about various "
              "components of the Mojo support in LLDB.");
  help.AddCommand("repl", new CommandREPLHelp(),
                  "Show a help message about the Mojo REPL.", "mojo help repl");
}
