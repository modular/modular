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

#ifndef MLRT_ASYNCRT_FORWARD_DECLS_H
#define MLRT_ASYNCRT_FORWARD_DECLS_H

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::MLRT {
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

} // namespace M::MLRT

//===----------------------------------------------------------------------===//
// Using Declarations
//===----------------------------------------------------------------------===//

namespace M {
// AsyncRT/Support Declarations
using MLRT::Chain;
using MLRT::EncodedDiagnostic;
using MLRT::EncodedLocation;
using MLRT::LocationDecoder;

// AsyncRT/Runtime Declarations
using MLRT::Allocator;
using MLRT::AnyAsyncValueRef;
using MLRT::AsyncValue;
using MLRT::AsyncValueRef;
using MLRT::Runtime;
} // namespace M

#endif // MLRT_ASYNCRT_FORWARD_DECLS_H
