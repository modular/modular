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
#include "llvm/ADT/SmallPtrSet.h"

namespace M::KGEN {
/// The definition of a parameter. The parameter definition contains its value
/// and the operation which contains the value attribute. Not all declared
/// parameters have definitions. Input parameters to a function, for example,
/// have no definition within the function, and are treated as leaves.
struct ParamDefinition {
  /// The value of the parameter, if it has a resolved one.
  Attribute value;
  /// The index of the parameter into the operation's result parameters.
  std::optional<ssize_t> index;
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
  /// the operation that declares the parameter.
  Operation *declOp;
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
  ParameterUseDefGraph(DeclInterface scope) : scope(scope) {}

  /// Map of parameter name to its declaration.
  DenseMap<StringAttr, ParamDeclaration> decls;
  /// Map of parameter name to its definition.
  DenseMap<StringAttr, ParamDefinition> defs;

  /// The scope at which this graph is computed.
  DeclInterface scope;

  /// A list of parametric operations. These are the operations that must be
  /// concretized by the elaborator once all parameters in the scope have been
  /// computed to simple constant values.
  std::vector<Operation *> paramOps;

  /// A list of all parameters defined within the scope.
  SmallVector<StringAttr> params;

  /// These are the parameter uses in the current scope that were captured from
  /// a higher scope.
  SmallPtrSet<ParamDeclRefAttr, 8> usesFromAbove;

  /// Track the operations that reference parameters. Use this information to
  /// diagnose references to parameters without declarations.
  DenseMap<Operation *, SmallVector<ParamDeclRefAttr>> opUses;

  /// A list of nested parameter scopes.
  SmallVector<DeclInterface> nestedDecls;

  /// A map of nested scopes to their use-def graph.
  DenseMap<DeclInterface, ParameterUseDefGraph> nestedScopes;

  /// Compute the parameter declarations, definitions, and uses within the
  /// provided parameter declaration scope. If the the root scope is not
  /// isolated from above, the use-def graph expects to be primed with the
  /// parent scope's declarations before this function is called.
  void calculate();

  /// Verify the validity of the parameter declarations, uses, and definitions
  /// within the current scope.
  LogicalResult verify(SymbolTableCollection &symtab);

private:
  /// Calculate the parameter use-def graph and perform verification if a symbol
  /// table is provided.
  LogicalResult calculateOrVerify(ModuleOp module,
                                  SymbolTableCollection *symtab);
};
} // namespace M::KGEN

#endif // KGEN_KGENPARAMETERS_H
