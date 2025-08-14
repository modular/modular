//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include <string>

void M::replaceAll(std::string &str, StringRef oldStr, StringRef newStr) {
  size_t pos = 0;
  size_t oldSize = oldStr.size();

  if (oldSize == 0)
    return;

  while ((pos = str.find(oldStr, pos)) != StringRef::npos) {
    str.replace(pos, oldSize, newStr);
    pos += newStr.size();
  }
}
