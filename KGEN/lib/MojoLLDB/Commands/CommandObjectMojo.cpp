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
    return true;
  }
};

} // namespace

void M::KGEN::Mojo::registerMojoCommands(SBDebugger debugger) {
  SBCommandInterpreter interpreter = debugger.GetCommandInterpreter();
  SBCommand root = interpreter.AddMultiwordCommand(
      "mojo", "Commands related to the Mojo language support.");
  SBCommand help = root.AddMultiwordCommand(
      "help", "Display help information about various "
              "components of the Mojo support in LLDB.");
  help.AddCommand("repl", new CommandREPLHelp(), "mojo help repl");
  help.AddCommand("debug", new CommandDebugHelp(), "mojo help debug");
}
