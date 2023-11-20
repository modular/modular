//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_ELABORATOR_H
#define KGEN_ELABORATOR_ELABORATOR_H

#include "Cache/CacheDialect/CachedTransform.h"
#include "IREvaluator.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "LLCL/CompilerSupport/AsyncSideEffectMap.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/Threading/Shared.h"
#include "Support/Threading/ThreadLocalCache.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// mangleParameterValues
//===----------------------------------------------------------------------===//

/// This returns a name to use when the specified generator is specialized
/// with the specified input parameters.
std::string mangleParameterValues(GeneratorOp generator,
                                  ArrayRef<TypedAttr> inputParamValues);

//===----------------------------------------------------------------------===//
// Elaborator
//===----------------------------------------------------------------------===//

using EvaluatorExecutorFnRef = function_ref<ErrorOr<ElaboratorSearchFn>(
    FuncOp, const SymbolTable &, TargetInfoAttr, ArrayRef<FuncOp>)>;
using ElaboratorCompileAsmFnRef = function_ref<ErrorOr<CrossDeviceFunction>(
    GeneratorOp, SymbolConstantAttr, StringAttr, const SymbolTable &,
    TargetInfoAttr, EmissionKind)>;

class Elaborator {
public:
  /// Enumeration of the compile assembly format.
  enum ASMFormat : uint8_t { ASM, LLVM };

  /// Initialize the elaborator and its symbol table.
  Elaborator(SymbolTable &symtab, TargetInfoAttr target,
             const ElaborateGeneratorsOptions &config)
      : symtab(symtab), target(target), config(config) {}

  virtual ~Elaborator() = default;

  /// Look up the callee symbol. If it's a FuncOp, return it. Otherwise,
  /// elaborate the generator or interface and return the first concrete
  /// implementation. Return none if the specialization is not ready yet.
  virtual std::optional<ErrorTreeOr<FuncOp>>
  getConcreteFunction(ImplNode *parent, Location loc,
                      FlatSymbolRefAttr symbolRef,
                      ArrayRef<TypedAttr> paramValues) = 0;

  /// Get all the concrete functions for the given symbol. If the symbol is a
  /// function already, append it to the list and move on, otherwise,
  /// elaborate it and append all the concrete implementations.
  virtual std::optional<ErrorTreeOrSuccess> getAllConcreteFunctions(
      ImplNode *parent, Location loc, FlatSymbolRefAttr symbolRef,
      ArrayRef<TypedAttr> paramValues, std::vector<FuncOp> &funcs) = 0;

  /// Get the functor for compiling a generator to assembly.
  virtual ElaboratorCompileAsmFnRef
  getCompileAsmFn(ASMFormat format = ASMFormat::ASM) const = 0;

  /// Add an owned function operation that should be appended to the module at
  /// the end of elaboration. This is where generated functions during
  /// elaboration should go.
  virtual void addDeferredFunction(OwningOpRef<FuncOp> func) = 0;

  /// Get the symbol table associated with this instance of the elaborator.
  Shared<SymbolTable &> &getSymbolTable() { return symtab; }
  /// Get the target associated with this instance of the elaborator.
  TargetInfoAttr getTarget() const { return target; }
  /// Get the elaborator config.
  const ElaborateGeneratorsOptions &getOptions() const { return config; }

protected:
  /// This symbol table allows efficient lookups across the module.
  Shared<SymbolTable &> symtab;

  /// The target we are compiling code for.
  TargetInfoAttr target;

  /// The elaborator config.
  ElaborateGeneratorsOptions config;
};

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_ELABORATOR_H
