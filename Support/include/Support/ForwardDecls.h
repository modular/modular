//===- Support/ForwardDecls.h ---------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file forward declares Support types in a canonical place.  This avoids
// scattering forward declarations throughout the codebase.
//
// This only covers the widely used types, not esoteric things like
// AlignedAlloc and ConcatenationTree.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_FORWARD_DECLS_H
#define SUPPORT_FORWARD_DECLS_H

namespace M {
class DType;
class ErrorOrSuccess;
template <typename T>
class ErrorOr;
} // namespace M

#endif // SUPPORT_FORWARD_DECLS_H
