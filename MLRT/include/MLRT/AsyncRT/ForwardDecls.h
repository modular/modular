//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file forward declares AsyncRT types in a canonical place and imports
// them into the Modular M namespace.  This avoids scattering forward
// declarations throughout the codebase.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_FORWARD_DECLS_H
#define ASYNCRT_FORWARD_DECLS_H

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::AsyncRT {
// AsyncRT/Support Declarations
class Chain;
class EncodedLocation;
class EncodedDiagnostic;
class LocationDecoder;

// AsyncRT/Runtime Declarations
class Allocator;
class AsyncValue;
class AnyAsyncValueRef;
template <typename T>
class AsyncValueRef;
class Runtime;

} // namespace M::AsyncRT

//===----------------------------------------------------------------------===//
// Using Declarations
//===----------------------------------------------------------------------===//

namespace M {
// AsyncRT/Support Declarations
using AsyncRT::Chain;
using AsyncRT::EncodedDiagnostic;
using AsyncRT::EncodedLocation;
using AsyncRT::LocationDecoder;

// AsyncRT/Runtime Declarations
using AsyncRT::Allocator;
using AsyncRT::AnyAsyncValueRef;
using AsyncRT::AsyncValue;
using AsyncRT::AsyncValueRef;
using AsyncRT::Runtime;
} // namespace M

#endif // ASYNCRT_FORWARD_DECLS_H
