//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_LANGUAGE_MOJODYNAMICVECTORTYPEFORMATTER_H
#define KGEN_LIB_MOJOLLDB_LANGUAGE_MOJODYNAMICVECTORTYPEFORMATTER_H

#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/lldb-forward.h"

namespace M::KGEN::Mojo {
class MojoDynamicVectorSyntheticFrontEnd
    : public lldb_private::SyntheticChildrenFrontEnd {
public:
  MojoDynamicVectorSyntheticFrontEnd(const lldb::ValueObjectSP &backend);

  ~MojoDynamicVectorSyntheticFrontEnd() override = default;

  size_t CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override;

  bool Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(lldb_private::ConstString name) override;

  /// Parse the given `ValueObject` representing a DynamicVector.
  ///
  /// Return a pair `<data pointer, size>`, where `data pointer` represents
  /// the start of the underlying data, and `size` represents the number of
  /// entries. If `size` is 0, then the data pointer might point to an invalid
  /// address.
  /// Otherwise, if `size` is larger than 0, the data pointer points to some
  /// address.
  ///
  /// This function returns null if it was not possible to read some of these
  /// fields, or if the invariants mentioned above don't hold.
  static std::optional<std::pair<lldb::ValueObjectSP, size_t>>
  parseDynamicVector(lldb::ValueObjectSP valobj);

private:
  lldb::addr_t start;
  size_t size;
  lldb_private::CompilerType elementType;
  uint64_t elementSize;
};

lldb_private::SyntheticChildrenFrontEnd *
mojoDynamicVectorSyntheticFrontEndCreator(lldb_private::CXXSyntheticChildren *,
                                          const lldb::ValueObjectSP &valobjSP);
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_MOJODYNAMICVECTORTYPEFORMATTER_H
