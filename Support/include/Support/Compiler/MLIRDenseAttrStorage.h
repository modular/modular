//===- Support/Compiler/MLIRDenseAttrStorage.h ----------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_MLIRDENSEATTRSTORAGE_H
#define SUPPORT_COMPILER_MLIRDENSEATTRSTORAGE_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M {
/// Returns true if the an array with the given number of elements is
/// sufficiently large that out-of-line storage should be used. This indicates
/// to the caller that the data is big enough to treat specially, e.g. that it
/// shouldn't be stored in the MLIRContext, folded unconditionally, etc.
bool shouldUseOutOfLineAttrStorage(size_t numElements);
} // namespace M

#endif // SUPPORT_COMPILER_MLIRDENSEATTRSTORAGE_H
