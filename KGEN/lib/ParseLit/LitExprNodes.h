//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides declarations for various expression nodes when in
// syntactic (not yet type checked) form.
//
// Expressions are parsed with a two-phase approach.  The first phase pulls out
// the syntactic structure of the expression, whereas the second pass does type
// checking and IR generation.
//
// These classes are formed in the first pass, owned by a bump pointer allocator
// whose lifetime matches the ExprParser class.  These nodes are not allowed to
// own resources because their destructors are never run.
//
//===----------------------------------------------------------------------===//

#ifndef LIT_EXPR_NODES_H
#define LIT_EXPR_NODES_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN::LIT {
using llvm::SMLoc;

/// Base class for all expression nodes.  Note that these nodes are not allowed
/// to own memory since they are bump pointer allocated and their destructors
/// are never run.
struct ExprNode {
  // This indicates the subclass.
  enum Kind {
    error,      // `
    intLiteral, // 42
    declRef,    // x
    call,       // thing(a, b)
  } const kind;

  ExprNode(Kind kind) : kind(kind) {}
  virtual ~ExprNode() { assert(0 && "never called"); }

  virtual SMLoc getLoc() const = 0;
};

/// This node is created to represent erroneous parses, but the diagnostic has
/// already been emitted.
struct ErrorNode : public ExprNode {
  ErrorNode(SMLoc loc) : ExprNode(error), loc(loc) {}

  SMLoc getLoc() const override { return loc; }
  SMLoc loc;
};

struct IntLiteralNode : public ExprNode {
  IntLiteralNode(StringRef spelling)
      : ExprNode(intLiteral), spelling(spelling) {}

  SMLoc getLoc() const override {
    return SMLoc::getFromPointer(spelling.data());
  }
  StringRef spelling;
};

struct DeclRefNode : public ExprNode {
  DeclRefNode(StringRef spelling) : ExprNode(declRef), spelling(spelling) {}

  SMLoc getLoc() const override {
    return SMLoc::getFromPointer(spelling.data());
  }
  StringRef spelling;
};

struct CallNode : public ExprNode {
  CallNode(ExprNode *callee, SMLoc lparenLoc, ArrayRef<ExprNode *> args)
      : ExprNode(call), callee(callee), lparenLoc(lparenLoc), args(args) {}

  SMLoc getLoc() const override { return lparenLoc; }

  ExprNode *callee;
  SMLoc lparenLoc;
  ArrayRef<ExprNode *> args;
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPR_NODES_H
