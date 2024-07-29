//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// AST representation for a declaration.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_TYPECHECKSCOPEINFO_H
#define KGEN_MOJOPARSER_TYPECHECKSCOPEINFO_H

namespace M::KGEN::LIT {
class ASTDecl;
class SharedState;

/// This struct is a common combination of information that is used when type
/// checking expressions.
struct TypeCheckScopeInfo {
  /// This is the declaration that we do name lookup against.
  ASTDecl &declScope;

  /// This is the shared state for the entire parser.
  SharedState &shared;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_TYPECHECKSCOPEINFO_H