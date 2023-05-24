//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectMojo.h"
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

//===----------------------------------------------------------------------===//
// CommandDumpLogs: dump-logs
//===----------------------------------------------------------------------===//
class CommandDumpLogs : public SBCommandPluginInterface {
public:
  bool DoExecute(SBDebugger debugger, char **command,
                 SBCommandReturnObject &result) override;

private:
  MojoTypeSystem *getMojoTypeSystem(SBDebugger &debugger,
                                    SBCommandReturnObject &result);
};
} // namespace

//===----------------------------------------------------------------------===//
// CommandHelp: help
//===----------------------------------------------------------------------===//
bool CommandHelp::DoExecute(SBDebugger debugger, char **command,
                            SBCommandReturnObject &result) {
  result.Printf("To be filled.\n");
  return true;
}

//===----------------------------------------------------------------------===//
// CommandDumpLogs: dump-logs
//===----------------------------------------------------------------------===//
bool CommandDumpLogs::DoExecute(SBDebugger debugger, char **command,
                                SBCommandReturnObject &result) {
  if (MojoTypeSystem *typeSystem = getMojoTypeSystem(debugger, result)) {
    typeSystem->flushIRDumpAndDebugLog();
    return true;
  }
  return false;
}

MojoTypeSystem *
CommandDumpLogs::getMojoTypeSystem(SBDebugger &debugger,
                                   SBCommandReturnObject &result) {
  SBTarget sbTarget = debugger.GetSelectedTarget();
  if (!sbTarget.IsValid()) {
    result.SetError("missing target.");
    return nullptr;
  }

  TargetSP target = SBTargetUtils::getSP(sbTarget);
  auto typeSystemOr =
      target->GetScratchTypeSystemForLanguage(lldb::eLanguageTypeMojo);
  if (!typeSystemOr) {
    result.SetError(llvm::toString(typeSystemOr.takeError()).c_str());
    return nullptr;
  }

  if (auto typeSystem = llvm::cast<MojoTypeSystem>(typeSystemOr.get().get())) {
    return typeSystem;
  } else {
    result.SetError("must be able to get the mojo type system");
    return nullptr;
  }
}

void M::KGEN::Mojo::registerMojoCommands(SBDebugger debugger) {
  SBCommandInterpreter interpreter = debugger.GetCommandInterpreter();
  SBCommand root = interpreter.AddMultiwordCommand(
      "mojo", "Commands related to the Mojo language support.");
  root.AddCommand("help", new CommandHelp(), "Display help information.");
  root.AddCommand("dump-logs", new CommandDumpLogs(),
                  "Dump the most recent unflushed development logs.");
}
