//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectMojo.h"
#include "../REPL/MojoREPL.h"
#include "../ScriptingBridge/SBClassUtils.h"
#include "../TypeSystem/MojoTypeSystem.h"
#include "lldb/Target/Target.h"

using namespace M;
using namespace M::KGEN::Mojo;
using namespace lldb;

namespace {
//===----------------------------------------------------------------------===//
// CommandHelp: help
//===----------------------------------------------------------------------===//
class CommandHelp : public SBCommandPluginInterface {
public:
  bool DoExecute(SBDebugger debugger, char **command,
                 SBCommandReturnObject &result) override;
};

} // namespace

bool CommandHelp::DoExecute(SBDebugger debugger, char **command,
                            SBCommandReturnObject &result) {
  result.AppendMessage(MojoREPL::GetHelpPrologue());
  return true;
}

void M::KGEN::Mojo::registerMojoCommands(SBDebugger debugger) {
  SBCommandInterpreter interpreter = debugger.GetCommandInterpreter();
  SBCommand root = interpreter.AddMultiwordCommand(
      "mojo", "Commands related to the Mojo language support.");
  root.AddCommand("help", new CommandHelp(), "Display help information.");
}
