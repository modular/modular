//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOWRAPPINGTYPESYNTHETICFRONTEND_H
#define KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOWRAPPINGTYPESYNTHETICFRONTEND_H

#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/lldb-forward.h"

namespace M::KGEN::Mojo {
/// Synthetic type front end that effectively replaces a value with one of its
/// children based on a path of indices, e.g. if the path is {1, 0}, then the
/// effective value will be `originalValue[1][0]`.
/// It is also able to create a synthetic version of the effective value, unlike
/// LLDB's DummySyntheticFrontEnd.
class MojoWrappingTypeSyntheticFrontEnd
    : public lldb_private::SyntheticChildrenFrontEnd {
public:
  MojoWrappingTypeSyntheticFrontEnd(lldb_private::ValueObject &backend,
                                    llvm::ArrayRef<size_t> path = {0})
      : SyntheticChildrenFrontEnd(backend), path(path) {}

  ~MojoWrappingTypeSyntheticFrontEnd() override = default;

  size_t CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override;

  size_t GetIndexOfChildWithName(lldb_private::ConstString name) override;

  bool Update() override;

  lldb::ValueObjectSP GetSyntheticValue() override;

  bool MightHaveChildren() override;

  lldb_private::ConstString GetSyntheticTypeName() override;

private:
  /// Get the synthetic value if available, or the original value otherwise.
  lldb::ValueObjectSP getEffectiveValue();

  MojoWrappingTypeSyntheticFrontEnd(const MojoWrappingTypeSyntheticFrontEnd &) =
      delete;
  const MojoWrappingTypeSyntheticFrontEnd &
  operator=(const MojoWrappingTypeSyntheticFrontEnd &) = delete;

  lldb::ValueObjectSP wrappedValue;
  std::vector<size_t> path;
};
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOWRAPPINGTYPESYNTHETICFRONTEND_H
