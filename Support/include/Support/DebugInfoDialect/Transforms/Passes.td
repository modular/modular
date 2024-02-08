//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// DebugInfoSnapshot
//===----------------------------------------------------------------------===//

def DebugInfoSnapshot : Pass<"debuginfo-snapshot"> {
  let summary = "Snapshot debug information for the current IR.";
  let description = [{
    This pass allows for generating synthetic debug information from the IR
    during any stage of compilation, by snapshotting the IR to a file and using
    that file to generate information for the operations. This overwrites any
    existing debug information within the IR.

    An optional filename may be provided to use for the snapshot, otherwise
    an intermediate file is generated.
  }];

  let options = [
    Option<"emissionKind", "emission", "EmissionKind",
           /*default=*/"EmissionKind::Full",
           "The kind of debug info emission">,
    Option<"filename", "filename", "std::string", /*default=*/"",
           "The filename to print the generated IR">,
    Option<"debugInfoLanguage", "debug-info-language",
           "llvm::dwarf::SourceLanguage", "llvm::dwarf::DW_LANG_Mojo",
           "The DWARF language to specify in the debug info, either `C` or "
           "`Mojo`. Defaults to `Mojo`.",
           "llvm::cl::values(clEnumValN(llvm::dwarf::DW_LANG_C, \"C\", \"C language.\"),"
           "clEnumValN(llvm::dwarf::DW_LANG_Mojo, \"Mojo\", \"Mojo language\"))">,
  ];
  let dependentDialects = ["DebugInfo::DebugInfoDialect"];
}

//===----------------------------------------------------------------------===//
// DebugInfoStrip
//===----------------------------------------------------------------------===//

def DebugInfoStrip : Pass<"debuginfo-strip"> {
  let summary = "Strip debug information for the current IR.";
  let description = [{
    This pass strips all debuginfo dialect information from the current IR, but
    leaves normal location information in place (to still allow proper
    verification and error reporting).
  }];
  let options = [
    Option<"preserveLineTables", "preserveLineTables", "bool",
           /*default=*/"false",
           "Don't strip information used for line table generation.">,
  ];
}
