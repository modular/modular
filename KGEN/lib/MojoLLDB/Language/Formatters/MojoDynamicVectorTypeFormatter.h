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
  MojoDynamicVectorSyntheticFrontEnd(lldb::ValueObjectSP backend);

  ~MojoDynamicVectorSyntheticFrontEnd() override = default;

  size_t CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(size_t idx) override;

  bool Update() override;

  bool MightHaveChildren() override;

  size_t GetIndexOfChildWithName(lldb_private::ConstString name) override;

private:
  lldb::addr_t start;
  size_t size;
  lldb_private::CompilerType elementType;
  uint64_t elementSize;
};

lldb_private::SyntheticChildrenFrontEnd *
MojoDynamicVectorSyntheticFrontEndCreator(lldb_private::CXXSyntheticChildren *,
                                          const lldb::ValueObjectSP &valobjSP);
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_MOJODYNAMICVECTORTYPEFORMATTER_H
