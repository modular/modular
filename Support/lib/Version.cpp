//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Version.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <tuple>

using namespace M;

ErrorOr<Version> Version::parse(StringRef str) {
  // Split the string on '.'. At most, we want 3 components because we'll parse
  // the last bit ourselves.
  SmallVector<StringRef> v;
  str.split(v, '.', /*MaxSplit=*/2, /*KeepEmpty=*/false);
  if (v.size() != 3) {
    return Error("expected exactly 3 components when split by dots, got: " +
                 str);
  }

  Version out;
  if (v[0].getAsInteger(10, out.major))
    return Error("major version must be an integer, got: " + v[0]);
  if (v[1].getAsInteger(10, out.minor))
    return Error("minor version must be an integer, got: " + v[1]);

  // Consume any integer that might be at the start of v[2].
  if (v[2].consumeInteger(10, out.patch))
    return Error("patch version must be an integer, got: " + v[2]);

  // If there's nothing else in the string, we're done.
  if (v[2].empty())
    return out;

  // Consume the hyphen that must be after the number.
  if (!v[2].consume_front("-")) {
    return Error("patch was not hyphen-separated, got: " + v[2]);
  }

  // The rest goes into the label.
  out.label = v[2].str();

  return out;
}

bool Version::operator<(const Version &other) const {
  if (major < other.major)
    return true;
  if (major > other.major)
    return false;
  if (minor < other.minor)
    return true;
  if (minor > other.minor)
    return false;
  if (patch < other.patch)
    return true;
  if (patch > other.patch)
    return false;
  // If we have a label and the other doesn't, then the other is greater.
  if (!label.empty() && other.label.empty())
    return true;
  if (label.empty() && !other.label.empty())
    return false;
  // Precedence is determined by comparing dot separated identifiers from left
  // to right until a difference is found.
  SmallVector<StringRef> ourLabels, otherLabels;
  StringRef(label).split(ourLabels, '.');
  StringRef(other.label).split(otherLabels, '.');
  //  1. Identifiers consisting of only digits are compared numerically
  //  2. Identifiers with letters or hyphens are compared lexically in ASCII
  //  sort order.
  //  3. Numeric identifiers always have lower precedence than non-numeric
  //  identifiers.
  //  4. A larger set of pre-release fields has a higher precedence (larger
  //  than) than a smaller set, if all preceding identifiers are equal.
  for (auto labels : llvm::zip(ourLabels, otherLabels)) {
    StringRef ourLabel, otherLabel;
    std::tie(ourLabel, otherLabel) = labels;

    size_t ourNum, otherNum;
    if (!ourLabel.getAsInteger(10, ourNum)) {
      // If the other label is *also* a number, return whichever is greater.
      if (!otherLabel.getAsInteger(10, otherNum))
        return ourNum < otherNum;
      else // Numeric identifiers have lower precedence.
        return true;
    }

    // Letters or hyphens means ASCII sort order.
    if (ourLabel < otherLabel)
      return true;
    if (ourLabel > otherLabel)
      return false;
  }

  // Finally, larger set of pre-release fields has higher precedence.
  if (ourLabels.size() < otherLabels.size())
    return true;

  // All comparisons failed, we are >= the other one.
  return false;
}

std::string Version::toString() const {
  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *this;
  return out;
}

llvm::raw_ostream &M::operator<<(llvm::raw_ostream &os, const Version &other) {
  os << other.getMajor() << "." << other.getMinor() << "." << other.getPatch();
  if (StringRef label = other.getLabel(); !label.empty())
    os << "-" << label;

  return os;
}
