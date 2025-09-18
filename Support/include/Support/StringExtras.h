//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_STRINGEXTRAS_H
#define SUPPORT_STRINGEXTRAS_H

#include "Support/LLVMForwardDecls.h"

namespace M {

/// Replace all occurrences of `oldStr` with `newStr` in `str`. Returns a new
/// string with the replacements. The replacement happens eagerly, from the
/// beginning of the string to the end (i.e. from lower indices to higher
/// indices).
void replaceAll(std::string &str, StringRef oldStr, StringRef newStr);

/// Pretty prints the memory in a human readable form (e.g. 1024 is
// printed as 1KB).
std::string humanMemorySize(size_t size);

} // namespace M

#endif // SUPPORT_STRINGEXTRAS_H
