//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines an IPInt class, which is a wrapper around APInt to
// represent (memory-bounded) infinite precision integers.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_IPINT_H
#define SUPPORT_IPINT_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"

namespace M {

/// IPInt is a wrapper around APInt to represent infinite precision integers.
/// The main motivation to write this was that APInts can't be compared for
/// equality when they have different bit widths (it instead raises an assertion
/// error).  But additionally this defines standard arithmetic operations on
/// infinite precision integers.
class IPInt {
public:
  /// The wrapped APInt is normalized to use the minimum number of bits so that
  /// equality testing works.
  IPInt(const llvm::APInt val) : val(val.trunc(val.getSignificantBits())){};
  IPInt(const IPInt &val) : val(val.val){};
  IPInt() : val(){};

  const llvm::APInt &getAPInt() const { return val; }

  IPInt &operator=(const IPInt &RHS) {
    val = RHS.getAPInt();
    return *this;
  }

  bool operator==(const IPInt &RHS) const;
  bool operator!=(const IPInt &RHS) const;
  bool operator<(const IPInt &RHS) const;
  bool operator<=(const IPInt &RHS) const;
  bool operator>(const IPInt &RHS) const;
  bool operator>=(const IPInt &RHS) const;
  IPInt operator+(const IPInt &RHS) const;
  IPInt operator-(const IPInt &RHS) const;
  IPInt operator*(const IPInt &RHS) const;
  IPInt operator/(const IPInt &RHS) const;
  IPInt operator%(const IPInt &RHS) const;
  IPInt operator<<(const IPInt &RHS) const;
  IPInt operator>>(const IPInt &RHS) const;
  IPInt operator&(const IPInt &RHS) const;
  IPInt operator|(const IPInt &RHS) const;
  IPInt operator^(const IPInt &RHS) const;
  IPInt pow(const IPInt &RHS) const;

  friend llvm::hash_code hash_value(const IPInt &Arg);

private:
  enum class BinOp {
    kAdd,
    kSub,
    kMul,
    kDiv,
    kMod,
    kLshift,
    kRshift,
    kAnd,
    kOr,
    kXor,
  };
  enum class CmpOp {
    kSgt,
    kSge,
    kSlt,
    kSle,
  };

  IPInt binop(const IPInt &RHS, IPInt::BinOp whichOp) const;
  bool cmp(const IPInt &RHS, IPInt::CmpOp whichOp) const;

  llvm::APInt val;
};

} // namespace M

#endif // SUPPORT_IPINT_H
