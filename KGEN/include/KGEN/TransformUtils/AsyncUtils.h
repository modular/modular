//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TRANSFORMUTILS_ASYNCUTILS_H
#define KGEN_TRANSFORMUTILS_ASYNCUTILS_H

namespace M::KGEN {

/// An AsyncContinuationField represents a slot in the continuation data
/// structure of a coroutine. Multiple passes depend on the layout of this data
/// structure so we store its definition in a shared location.
enum AsyncContinuationField {
  State = 0,
  ResumeFunction = 1,
  CallbackFn = 2,
  ClosureState = 3,
  ErrorSlot = 4,
  ResultSlot = 5,
  Promise = 6,
  Frame = 7
};

} // namespace M::KGEN

#endif // KGEN_TRANSFORMUTILS_ASYNCUTILS_H
