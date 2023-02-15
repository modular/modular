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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// ParameterCollector
//===----------------------------------------------------------------------===//

/// Visit the type and all its sub-elements and collect all parameter
/// references at the scope of the type.
void collectParameterUsesFrom(Type type,
                              SmallVectorImpl<ParamDeclRefAttr> &uses);
/// Visit the attribute and all its sub-elements and collect all parameter
/// references at the scope of the attribute.
void collectParameterUsesFrom(Attribute attr,
                              SmallVectorImpl<ParamDeclRefAttr> &uses);

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

  /// A map of nested scopes to their use-def graph.
  DenseMap<Region *, ParameterUseDefGraph> nestedScopes;

  /// Compute the parameter declarations, definitions, and uses within the
  /// provided parameter declaration scope. If the the root scope is not
  /// isolated from above, the use-def graph expects to be primed with the
  /// parent scope's declarations before this function is called.
  void calculate();

  /// Verify the validity of the parameter declarations, uses, and definitions
  /// within the current scope.
  LogicalResult verify(SymbolTableCollection &symtab);

  /// Copy this graph into a new instance, remapping all the operations using
  /// `map`.
  ParameterUseDefGraph copy(const IRMapping &map);

private:
  /// Calculate the parameter use-def graph and perform verification if a symbol
  /// table is provided.
  LogicalResult calculateOrVerify(ModuleOp module,
                                  SymbolTableCollection *symtab);
};

} // namespace M::KGEN

#endif // KGEN_KGENPARAMETERS_H
