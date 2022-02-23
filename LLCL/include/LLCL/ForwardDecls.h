//===- LLCL/ForwardDecls.h - Forward Declare LLCL Types ---------*- C++ -*-===//
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

// Support/RCRef.h
template <typename T>
class RCRef;

// Support/ReferenceCounted.h
template <typename SubClass>
class ReferenceCounted;

// LLCL/Runtime Declarations

// Runtime/Runtime.h
class Runtime;

} // end namespace LLCL

//===----------------------------------------------------------------------===//
// Using Declarations
//===----------------------------------------------------------------------===//

namespace M {
// LLCL/Support Declarations
using LLCL::RCRef;
using LLCL::ReferenceCounted;

// LLCL/Runtime Declarations
using LLCL::Runtime;
} // namespace M

#endif // SUPPORT_FORWARD_DECLS_H