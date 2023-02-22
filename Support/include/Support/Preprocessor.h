//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines common preprocessor methods.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PREPROCESSOR_H
#define SUPPORT_PREPROCESSOR_H

#define STRINGIFY_IMPL(X) #X
#define STRINGIFY(X) STRINGIFY_IMPL(X)

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define M_UNIQUE_NAME(base) CONCAT(base, __COUNTER__)

#endif // SUPPORT_PREPROCESSOR_H
