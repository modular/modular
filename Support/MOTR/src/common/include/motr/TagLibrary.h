//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_GUI_TAG_LIBRARY_H
#define MOTR_GUI_TAG_LIBRARY_H

#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>

#include "motr/Hash.h"
#include "motr/MString.h"
#include "motr/Tags.h"

namespace M::motr {

struct EventTreeNode;
struct TagLibrary {
  // using Ptr = std::unique_ptr<TagLibrary>;
  using Ptr = std::shared_ptr<TagLibrary>;

  TagLibrary();
  ~TagLibrary();
  TagLibrary(const TagLibrary &) = delete;
  TagLibrary &operator=(const TagLibrary &) = delete;
  TagLibrary(TagLibrary &&) = delete;
  TagLibrary &operator=(TagLibrary &&) = delete;

  bool hasTag(MString key) const;
  bool hasTagStr(MString key) const;
  bool hasTagInt(MString key) const;

  MString getMString(MString key) const;
  bool getString(MString key, std::string_view &value) const;
  std::string_view getString(MString key) const;
  std::optional<std::string_view> getOptionalString(MString key) const;
  MString setString(MString key, std::string_view value);

  bool getU64(MString key, uint64_t &value) const;
  uint64_t getU64(MString key) const; // get with default value 0
  std::optional<uint64_t> getOptionalU64(MString key) const;
  MString setU64(MString key, uint64_t value);

  using TagStrMap = std::unordered_map<MString, std::string>;
  using TagIntMap = std::unordered_map<MString, uint64_t>;

  size_t initFromEventTreeNode(const EventTreeNode &node);

  static Ptr create(const EventTreeNode &node);
  static Ptr pushContext(Ptr parent);

  uint64_t getVersion(MString key) const;

  TagStrMap tagStrMap;
  TagIntMap tagIntMap;

  TagIntMap tagVersionMap;

  void setLocalOnly(bool value) { localOnly = value; }
  bool isLocalOnly() const { return localOnly; }

private:
  std::weak_ptr<TagLibrary> parentContext;
  bool localOnly = false;
};

} // namespace M::motr

#endif // MOTR_GUI_TAG_LIBRARY_H
