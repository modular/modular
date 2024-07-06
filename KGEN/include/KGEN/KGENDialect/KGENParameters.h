//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains helper functions for working with KGEN parameter
// expressions and declarations.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENPARAMETERS_H
#define KGEN_KGENPARAMETERS_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/ParameterReplacer.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// IndexRefRemapper
//===----------------------------------------------------------------------===//

/// Utility class for remapping named parameter references to index references.
class IndexRefRemapper : public IndexParameterReplacer<IndexRefRemapper> {
public:
  /// Populate the remapper with named input and result parameters.
  IndexRefRemapper(ArrayRef<ParamDeclAttr> inputParams,
                   ArrayRef<ParamDeclAttr> resultParams, size_t offset = 0);

  /// Populate the remapper with the given named input parameters. If
  /// 'addOffset' is true, the underlying offset of references to root
  /// parameters will be incremented by the size of 'params'
  void populate(ArrayRef<ParamDeclAttr> params, bool isResult,
                bool addOffset = false);

private:
  // CRTP methods.
  Attribute tryReplace(Attribute attr, size_t depth);
  Type tryReplace(Type, size_t) { return {}; }
  friend class IndexParameterReplacer<IndexRefRemapper>;

  /// Mapping from parameter reference to an index and `isResult` flag.
  DenseMap<StringAttr, std::pair<size_t, bool>> mapping;
  /// The index offset of references to root input parameters.
  size_t offset;
};

//===----------------------------------------------------------------------===//
// IndexDepthAdjuster
//===----------------------------------------------------------------------===//

/// This class is used exclusively to adjust the depths of index references that
/// reference signatures outside the current scope.
class IndexDepthAdjuster : public IndexParameterReplacer<IndexDepthAdjuster> {
public:
  explicit IndexDepthAdjuster(int64_t adjustDepth) : adjustDepth(adjustDepth) {}

private:
  // CRTP methods.
  Attribute tryReplace(Attribute attr, size_t depth);
  Type tryReplace(Type, size_t) { return {}; }
  friend class IndexParameterReplacer<IndexDepthAdjuster>;

  /// Adjust the depth of index references when remapping.
  int64_t adjustDepth;
};

//===----------------------------------------------------------------------===//
// ParameterCollector
//===----------------------------------------------------------------------===//

class ParameterCollector {
public:
  /// The parameter collector contains a cache of parameter-less attributes and
  /// types that is valid throughout the lifetime of an MLIR context. This
  /// analysis allows the cache to be preserved across passes.
  struct Analysis {
    Analysis(Operation *op = nullptr) {}

    /// This analysis can never be invalid.
    bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
      return false;
    }

    /// Types and attributes contained in this map are known to have no
    /// parameter uses as sub-elements. They are mapped to whether there is an
    /// unresolved parameter operator in the sub-elements.
    DenseMap<const void *, bool> parameterLess;
  };

  /// Create a parameter collector with a collection cache.
  ParameterCollector(Analysis &cache) : cache(cache) {}

  virtual ~ParameterCollector() = default;

  /// Scan the specified attribute and its recursive uses, diagnosing incorrect
  /// parameter declarations and collecting parameter uses into `uses`.
  void collectUsesFromAttr(Attribute attr,
                           SmallVectorImpl<ParamDeclRefAttr> &uses,
                           bool &hasConstExpr);

  /// Scan the specified type and its recursive uses, diagnosing incorrect
  /// parameter declarations and collecting parameter uses into `uses`.
  void collectUsesFromType(Type type, SmallVectorImpl<ParamDeclRefAttr> &uses,
                           bool &hasConstExpr);

private:
  void collectUsesFromTypesImpl(Type type,
                                SmallVectorImpl<ParamDeclRefAttr> &uses,
                                bool &hasConstExpr);

  /// The first time we encounter an attribute with a reference to an
  /// out-of-line declaration, verify it.
  virtual void verifyRefAttr(DeclRefAttrInterface refAttr) {}

  /// When we encounter a StructType, check that its parameter bindings match
  /// the parameter declarations on the type declaration.
  virtual void verifyRefType(StructTypeInterface refType) {}

  /// Optionally perform verification and emit an error.
  virtual void
  maybeVerify(function_ref<LogicalResult(function_ref<InFlightDiagnostic()>)>
                  verifyFn) {}

  /// Attributes and types are memoized and exist in tree structures with reuse:
  /// naively scanning them can lead to exponential compile time behavior.  As
  /// such, we memoize the attributes and types we've already checked that we
  /// know have no parameters in them and whether the paramless attributes are
  /// constant parameter expressions.
  Analysis &cache;

  /// An internal stack of scoped parameter types representing the current
  /// nested signatures.
  SmallVector<ParameterScopeTypeInterface> signatures;
};

//===----------------------------------------------------------------------===//
// ParameterUseDefGraph
//===----------------------------------------------------------------------===//

/// The definition of a parameter. The parameter definition contains its value
/// and the operation which contains the value attribute. Not all declared
/// parameters have definitions. Input parameters to a function, for example,
/// have no definition within the function, and are treated as leaves.
struct ParamDefinition {
  /// If the expression that defines the parameter can be narrowed to a simple
  /// attribute, this field will contain that expression.
  Attribute value;
  /// The index of the parameter into the operation's result parameters. This is
  /// -1 for a parameter that is not a result parameter.
  ssize_t index = -1;
  /// The defining operation.
  Operation *defOp = nullptr;
  /// The dependent parameters of the definition.
  SmallVector<ParamDeclRefAttr> uses;
};

/// The declaration of a parameter. The parameter declaration contains the type
/// of the parameter and the operation that declares it. A parameter can be
/// declared and defined by different operations: a return parameter, for
/// example, is declared by the surrounding function but defined by its return
/// operation.
struct ParamDeclaration {
  /// The type of the parameter as it was declared.
  Type type;
  /// The operation that declares the parameter.
  Operation *declOp;
  /// The parent declaration scope.
  Region *scope;
};

/// This class defines the use-def graph for parameters. There are two types of
/// parameter uses: operations and parameter definitions. The use-def graph of
/// parameter declarations and definitions is of most interest: there can be
/// no cycles in this graph.
///
/// The elaborator must first resolve this graph by providing values for the
/// leaf nodes (input parameters) and computing all the parameter definition
/// expressions to a simple constant value. Then, all operations that use
/// parameters (in an attribute, type, or location) can be concretized in any
/// order.
struct ParameterUseDefGraph {
  ParameterUseDefGraph(Region &scope) : scope(&scope) {}
  ParameterUseDefGraph(Region *scope) : scope(scope) {}

  /// Map of parameter name to its declaration.
  DenseMap<StringAttr, ParamDeclaration> decls;
  /// Map of parameter name to its definition.
  DenseMap<StringAttr, ParamDefinition> defs;

  /// The scope at which this graph is computed.
  Region *scope;

  /// A list of parametric operations. These are the operations that must be
  /// concretized by the elaborator once all parameters in the scope have been
  /// computed to simple constant values.
  std::vector<Operation *> paramOps;

  /// A list of all parameters defined within the scope.
  SmallVector<StringAttr> params;

  /// These are the parameter uses in the current scope that were captured from
  /// a higher scope.
  llvm::SetVector<ParamDeclRefAttr> usesFromAbove;

  /// Track the operations that reference parameters. Use this information to
  /// diagnose references to parameters without declarations.
  llvm::MapVector<Operation *, SmallVector<ParamDeclRefAttr>> opUses;

  /// A list of nested parameter scopes.
  SmallVector<Region *> nestedDecls;

  /// A map of nested scopes to their use-def graph. Note that when calculating
  /// the use-def graph, the top-level use-def graph contains the mappings for
  /// ALL the nested scopes. The graphs of nested scopes must be looked up on
  /// the top-level graph.
  DenseMap<Region *, ParameterUseDefGraph> nestedScopes;

  /// Compute the parameter declarations, definitions, and uses within the
  /// provided parameter declaration scope. If the the root scope is not
  /// isolated from above, the use-def graph expects to be primed with the
  /// parent scope's declarations before this function is called.
  void calculate(ParameterCollector::Analysis &cache);

  /// Verify the validity of the parameter declarations, uses, and definitions
  /// within the current scope.
  LogicalResult verify(mlir::LockedSymbolTableCollection &symtab,
                       ParameterCollector::Analysis &cache);

  /// Copy this graph into a new instance, remapping all the operations using
  /// `map`.
  ParameterUseDefGraph copy(const IRMapping &map) const;

  /// Print the graph to llvm::errs().
  void dump() const;

  /// Disable implicit copying.
  ParameterUseDefGraph(const ParameterUseDefGraph &) = delete;
  ParameterUseDefGraph &operator=(const ParameterUseDefGraph &) = delete;
  ParameterUseDefGraph(ParameterUseDefGraph &&) = default;
  ParameterUseDefGraph &operator=(ParameterUseDefGraph &&) = default;

private:
  /// Calculate the parameter use-def graph and perform verification if a symbol
  /// table is provided.
  LogicalResult calculateOrVerify(ModuleOp module,
                                  mlir::LockedSymbolTableCollection *symtab,
                                  ParameterCollector::Analysis &cache);
};

} // namespace M::KGEN

#endif // KGEN_KGENPARAMETERS_H
