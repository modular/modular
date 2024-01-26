//===----------------------------------------------------------------------===//
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

namespace M::LLCL {
// LLCL/Support Declarations
class Chain;
class EncodedLocation;
class EncodedDiagnostic;
class LocationDecoder;

// LLCL/Runtime Declarations
class Allocator;
class AsyncValue;
class AnyAsyncValueRef;
template <typename T>
class AsyncValueRef;
class Runtime;

} // namespace M::LLCL

//===----------------------------------------------------------------------===//
// Using Declarations
//===----------------------------------------------------------------------===//

namespace M {
// LLCL/Support Declarations
using LLCL::Chain;
using LLCL::EncodedDiagnostic;
using LLCL::EncodedLocation;
using LLCL::LocationDecoder;

// LLCL/Runtime Declarations
using LLCL::Allocator;
using LLCL::AnyAsyncValueRef;
using LLCL::AsyncValue;
using LLCL::AsyncValueRef;
using LLCL::Runtime;
} // namespace M

#endif // LLCL_FORWARD_DECLS_H
