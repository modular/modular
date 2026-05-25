//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_DECLSIGNATUREPRINTER_H
#define KGEN_MOJOPARSER_DECLSIGNATUREPRINTER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

namespace M {
namespace KGEN {
namespace LIT {
class AliasDeclOp;
class ASTDecl;
class FnOp;
class SharedState;
class StructDeclOp;
} // namespace LIT
} // namespace KGEN

namespace KGEN {

/// Optional output ranges to capture during signature emission. Useful for
/// downstream consumers (e.g. mojo-doc) that need to highlight or navigate to
/// individual signature components. All offsets are measured from the start of
/// the underlying string buffer being written into.
struct SignatureOffsets {
  /// Half-open `[start, end)` byte ranges for each documented parameter, in
  /// emission order.
  llvm::SmallVectorImpl<std::pair<unsigned, unsigned>> *parameters = nullptr;
  /// Half-open `[start, end)` byte ranges for each documented argument, in
  /// emission order.
  llvm::SmallVectorImpl<std::pair<unsigned, unsigned>> *arguments = nullptr;
  /// Byte offset where the return-type clause begins. Set even when there is
  /// no return type, so callers can splice in extra text (e.g. ` raises`)
  /// just before the result.
  unsigned *returnTypeStart = nullptr;
};

/// Print a Mojo-syntax signature for the given function op.
///
/// No leading `def` keyword is emitted. If `contextDecl` is non-null, it
/// is installed as the current diagnostic decl context for the duration of the
/// call so dependent parameter references can be rendered with their source
/// names; otherwise dependent names may fall back to index references.
void printFunctionSignature(LIT::FnOp fnOp, LIT::SharedState &shared,
                            llvm::raw_string_ostream &os,
                            const LIT::ASTDecl *contextDecl = nullptr,
                            const SignatureOffsets &offsets = {});

/// Print a Mojo-syntax signature for the given struct op.
///
/// No leading `struct` keyword is emitted.
void printStructSignature(LIT::StructDeclOp structOp, LIT::SharedState &shared,
                          llvm::raw_string_ostream &os,
                          const LIT::ASTDecl *contextDecl = nullptr,
                          const SignatureOffsets &offsets = {});

/// Print a Mojo-syntax signature for the given alias op.
///
/// No leading `comptime` keyword is emitted.
void printAliasSignature(LIT::AliasDeclOp aliasOp, LIT::SharedState &shared,
                         llvm::raw_string_ostream &os,
                         const LIT::ASTDecl *contextDecl = nullptr,
                         const SignatureOffsets &offsets = {});

} // namespace KGEN
} // namespace M

#endif // KGEN_MOJOPARSER_DECLSIGNATUREPRINTER_H
