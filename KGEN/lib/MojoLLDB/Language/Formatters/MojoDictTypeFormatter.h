//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_LANGUAGE_MOJODICTTYPEFORMATTER_H
#define KGEN_LIB_MOJOLLDB_LANGUAGE_MOJODICTTYPEFORMATTER_H

#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/lldb-forward.h"
#include <vector>

namespace M::KGEN::Mojo {

/// Synthetic children front end for Mojo's Dict[K, V] type.
///
/// Dict uses a Swiss Table design with three arrays:
///   _ctrl  — one control byte per slot (0xFF=empty, 0x80=deleted, <0x80=live)
///   _slots — flat DictEntry[K,V,H] array indexed by slot number
///   _order — List[Int32] of slot indices in insertion order (may include
///             stale deleted-slot entries; live count is _len)
///
/// This front end walks _order, skips deleted slots via ctrl-byte checks, and
/// exposes exactly _len live DictEntry values as numbered children [0]..[N-1].
class MojoDictSyntheticFrontEnd
    : public lldb_private::SyntheticChildrenFrontEnd {
public:
  MojoDictSyntheticFrontEnd(const lldb::ValueObjectSP &backend);
  ~MojoDictSyntheticFrontEnd() override = default;

  llvm::Expected<uint32_t> CalculateNumChildren() override;
  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;
  lldb::ChildCacheState Update() override;
  bool MightHaveChildren() override;
  llvm::Expected<size_t>
  GetIndexOfChildWithName(lldb_private::ConstString name) override;

private:
  lldb::addr_t m_slotsAddr = 0;
  lldb_private::CompilerType m_entryType;
  uint64_t m_entrySize = 0;
  std::vector<uint32_t> m_liveSlots;
};

lldb_private::SyntheticChildrenFrontEnd *
mojoDictSyntheticFrontEndCreator(lldb_private::CXXSyntheticChildren *,
                                 const lldb::ValueObjectSP &valobjSP);

} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_MOJODICTTYPEFORMATTER_H
