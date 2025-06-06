//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_SCRIPTINGBRIDGE_SBCLASSUTILS_H
#define KGEN_LIB_MOJOLLDB_SCRIPTINGBRIDGE_SBCLASSUTILS_H

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBValue.h"

namespace M::KGEN::Mojo {

namespace Detail {
/// Class that exposes common utilities reated to SB classes. It should be used
/// via the concrete implementations that can be found below this class.
template <typename SBClass, typename SPClass>
class SBClassUtils : public SBClass {
public:
  /// Create a new instance of an SB class from the shared pointer of an
  /// lldb_private class.
  static SBClass create(const SPClass &privateObject) {
    return SBClassUtils(privateObject);
  }

  /// Get the lldb_private shared pointer wrapped by the given SB class.
  static SPClass getSP(const SBClass &publicObject) {
    return SBClassUtils(publicObject).GetSP();
  }

private:
  SBClassUtils(const SPClass &privateObject) : SBClass(privateObject) {}

  SBClassUtils(const SBClass &publicObject) : SBClass(publicObject) {}
};
} // namespace Detail

//===--------------------------------------------------------------------===//
// Concrete implementations
//===--------------------------------------------------------------------===//

class SBDebuggerUtils
    : public Detail::SBClassUtils<lldb::SBDebugger, lldb::DebuggerSP> {};

class SBTargetUtils
    : public Detail::SBClassUtils<lldb::SBTarget, lldb::TargetSP> {};

class SBValueUtils
    : public Detail::SBClassUtils<lldb::SBValue, lldb::ValueObjectSP> {};

} // namespace M::KGEN::Mojo
#endif // KGEN_LIB_MOJOLLDB_SCRIPTINGBRIDGE_SBCLASSUTILS_H
