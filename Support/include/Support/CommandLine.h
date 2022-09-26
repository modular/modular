//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file is a convenience wrapper around llvm/Support/CommandLine.h.  In
// addition to including it, this defines an `M::cl` namespace analogous to the
// `llvm::cl` namespace with the important types and functions imported.  This
// avoids having massively llvm::cl::opt sorts of qualifications in Modular
// code.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMMANDLINE_H
#define SUPPORT_COMMANDLINE_H

#include "llvm/Support/CommandLine.h"

namespace M {

namespace cl {

using alias = llvm::cl::alias;

template <class DataType, class StorageClass = bool,
          class ParserClass = llvm::cl::parser<DataType>>
using list = llvm::cl::list<DataType, StorageClass, ParserClass>;

template <class DataType, bool ExternalStorage = false,
          class ParserClass = llvm::cl::parser<DataType>>
using opt = llvm::cl::opt<DataType, ExternalStorage, ParserClass>;

using desc = llvm::cl::desc;
using value_desc = llvm::cl::value_desc;

template <class Ty>
inline llvm::cl::initializer<Ty> init(const Ty &Val) {
  return llvm::cl::init(Val);
}
template <typename... OptsTy>
inline llvm::cl::ValuesClass values(OptsTy... Options) {
  return llvm::cl::values(Options...);
}

} // namespace cl

} // namespace M

#endif
