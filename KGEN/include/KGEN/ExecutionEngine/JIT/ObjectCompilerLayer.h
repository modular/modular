//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef KGEN_EXECUTIONENGINE_JIT_OBJECTCOMPILERLAYER_H
#define KGEN_EXECUTIONENGINE_JIT_OBJECTCOMPILERLAYER_H

#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/ExecutionEngine/ExecutionEngine.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "MaterializationLayer.h"

namespace M::KGEN {
//===----------------------------------------------------------------------===//
// ObjectCompilerLayer
//===----------------------------------------------------------------------===//

/// Provide an ExecutionEngine layer for the ObjectCompiler. This simply wraps
/// up the ObjectCompiler in the correct APIs - under the hood it also defines a
/// MaterializationUnit and uses that to emit symbols on-demand.
class ObjectCompilerLayer : public MaterializationLayer {
public:
  ObjectCompilerLayer(std::unique_ptr<ObjectCompiler> objCompiler,
                      llvm::orc::ObjectLayer &base,
                      llvm::orc::ExecutionSession &sess,
                      const llvm::DataLayout &dl, AddToSearchOrderFn add);

  /// Add a module after KGEN pipeline to the JIT.
  ErrorOrSuccess add(StringRef libName, ModuleOp theModule);

  /// Emit a given module. This will immediately run the materialization.
  void emit(std::unique_ptr<llvm::orc::MaterializationResponsibility> mr,
            const SymbolTable &symtab, const ExportMap &exports);

  /// Provide access to the underlying ObjectCompiler so that users can call its
  /// methods directly if desired (for example, to emit asm or LLVM).
  ObjectCompiler &getRawCompiler() { return *objectCompiler; }

  static bool classof(const MaterializationLayer *layer) {
    return layer->getKind() == LayerKind::kObjectCompilerLayer;
  }

private:
  /// Emit a given module. This will immediately run the materialization.
  /// Returns errors rather than setting them on the materialization layer.
  ErrorOrSuccess emitImpl(llvm::orc::MaterializationResponsibility &mr,
                          const SymbolTable &symtab, const ExportMap &exports);

  /// Conform to the ORC's interface and return a map of the exported symbols.
  /// If the export map is empty, uses `getExportedSymbols` to infer them from
  /// the module.
  llvm::orc::MaterializationUnit::Interface
  getInterface(const SymbolTable &symtab, const ExportMap &exports);

  /// Provide an ObjectCompilerMaterializationUnit so that we can do codegen
  /// on-demand.
  class ObjectCompilerMaterializationUnit;

private:
  std::unique_ptr<ObjectCompiler> objectCompiler;
  llvm::orc::ObjectLayer &baseLayer;
};
} // namespace M::KGEN

#endif // KGEN_EXECUTIONENGINE_JIT_OBJECTCOMPILERLAYER_H
