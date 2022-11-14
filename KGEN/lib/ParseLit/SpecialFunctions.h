//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides information for working with 'special functions' in Lit
// like the __new__ function.
//
//===----------------------------------------------------------------------===//

#ifndef SPECIAL_FUNCTIONS_H
#define SPECIAL_FUNCTIONS_H

namespace M::KGEN::LIT {

enum class SpecialFunctionKind {
  // This is not a special function.  This enumerator should always have value
  // zero so it can be used as a false condition in an if.
  kNormal = 0,

  kInit = 1, //< __init__
  kNew = 2,  //< __new__
};

/// If this is a special function like __init__ return the enum that
/// identifies it, otherwise return kNormal.
SpecialFunctionKind getSpecialFunctionKind(StringRef name);

} // namespace M::KGEN::LIT

#endif // SPECIAL_FUNCTIONS_H
