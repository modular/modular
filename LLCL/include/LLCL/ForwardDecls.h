//===- LLCL/ForwardDecls.h ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file forward declares LLCL types in a canonical place and imports them
// into the Modular M namespace.  This avoids scattering forward declarations
// throughout the codebase.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_FORWARD_DECLS_H
#define LLCL_FORWARD_DECLS_H

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace LLCL {
// LLCL/Support Declarations
class Chain;

template <typename T>
class RCRef;
template <typename T>
RCRef<T> copyRCRef(T *ptr);
template <typename T>
RCRef<T> takeRCRef(T *ptr);

template <typename SubClass>
class ReferenceCounted;

class EncodedLocation;
class EncodedDiagnostic;
class LocationDecoder;

// LLCL/Runtime Declarations

class Allocator;
class AsyncValue;
template <typename T>
class AsyncValueRef;
class Runtime;

} // end namespace LLCL

//===----------------------------------------------------------------------===//
// Using Declarations
//===----------------------------------------------------------------------===//

namespace M {
// LLCL/Support Declarations
using LLCL::Chain;
using LLCL::copyRCRef;
using LLCL::EncodedDiagnostic;
using LLCL::EncodedLocation;
using LLCL::LocationDecoder;
using LLCL::RCRef;
using LLCL::ReferenceCounted;
using LLCL::takeRCRef;

// LLCL/Runtime Declarations
using LLCL::Allocator;
using LLCL::AsyncValue;
using LLCL::AsyncValueRef;
using LLCL::Runtime;
} // namespace M

#endif // LLCL_FORWARD_DECLS_H