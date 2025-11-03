//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_CLOPTIONUTILS_H
#define SUPPORT_COMPILER_CLOPTIONUTILS_H

namespace M {

// Register llvm::codegen::RegisterCodegenFlags flags.
// E.g. we want to use denormal-fp-math-f32
void registerCommandFlags();

} // namespace M

#endif // SUPPORT_COMPILER_CLOPTIONUTILS_H
