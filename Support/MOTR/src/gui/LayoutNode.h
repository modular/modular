//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef M_MOTR_GUI_LAYOUT_WINDOW_H
#define M_MOTR_GUI_LAYOUT_WINDOW_H

#include "Color.h"
#include "imgui.h"
#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "attr/DynamicValue.h"
#include "motr/Hash.h"
#include "motr/MString.h"
#include "nlohmann/json_fwd.hpp"

struct YGNode;
typedef struct YGNode *YGNodeRef;

namespace M::motr {
struct TagLibrary;
} // namespace M::motr

namespace M::motr::Gui {

// Forward declarations
struct LayoutNode;
struct WindowLayoutNode;
struct RefLayoutNode;
struct PlotLayoutNode;

enum class LayoutNodeType : int {
  Node,
  Window,
  Text,
  Image,
  Button,
  Plot,
  Ref,
};

using LayoutAttributes = std::unordered_map<MString, Attribute::DynamicValue,
                                            std::hash<M::motr::MString>>;

template <const char *_key_name, typename T>
struct NamedAttribute {
  static constexpr std::string_view key_sv{_key_name};
  static constexpr Hash::Value key_hash{_key_name};

  static const T &getValueFrom(const LayoutAttributes &attrs,
                               const T &defaultValue) {
    MString keyMstr{key_hash};
    auto it = attrs.find(keyMstr);
    if (it == attrs.end())
      return defaultValue;
    const Attribute::DynamicValue &dv = it->second;
    const T *p = std::get_if<T>(&dv.cache);
    if (!p)
      return defaultValue;
    return *p;
  }
};

template <typename T>
const LayoutAttributes &getAttributes(const T *node);

#define SCHEMA_DECLARE_NAMED_ATTRIBUTE(_KEY_NAME, _TYPE, _DEFAULT_VALUE)       \
  const _TYPE &get_##_KEY_NAME() const {                                       \
    static constexpr const char key[] = #_KEY_NAME;                            \
    static NamedAttribute<key, _TYPE> namedAttr;                               \
    auto &attrs = getAttributes(this);                                         \
    return namedAttr.getValueFrom(attrs, _DEFAULT_VALUE);                      \
  }

struct LayoutNodeSchema {
  /*
  SCHEMA_DECLARE_NAMED_ATTRIBUTE(position,                // name
                                 YGPositionType,          // type
                                 YGPositionTypeRelative); // default value
                                 */
  SCHEMA_DECLARE_NAMED_ATTRIBUTE(name, std::string, "");
  SCHEMA_DECLARE_NAMED_ATTRIBUTE(backgroundColor, Color::RGBA32,
                                 (Color::RGBA32{255, 255, 255, 255}));
};

struct LayoutNode : public std::enable_shared_from_this<LayoutNode>,
                    public LayoutNodeSchema {
  using Ptr = std::shared_ptr<LayoutNode>;

  static Ptr makeFromJsonStrView(const std::string_view &layout);
  static Ptr makeFromJson(const nlohmann::json &json);

  virtual ~LayoutNode();

  LayoutNode(const LayoutNode &other) = delete;
  LayoutNode &operator=(const LayoutNode &other) = delete;
  LayoutNode(LayoutNode &&other) noexcept = delete;
  LayoutNode &operator=(LayoutNode &&other) noexcept = delete;

  std::weak_ptr<LayoutNode> parent;
  std::vector<Ptr> children;

  size_t appendChild(Ptr child);
  Ptr getChildByName(std::string_view name) const;

  const Attribute::DynamicValue *getAttr(const MString &key) const;
  Attribute::DynamicValue *getAttr(const MString &key);
  bool copyAttrFrom(const LayoutNode &src, MString key);

  LayoutNodeType type = LayoutNodeType::Node;
  YGNodeRef node{nullptr};
  bool visible{true};
  std::string fmt;
  std::vector<std::string> args;

  LayoutAttributes attrs; // dynamic attributes

  virtual void relayout();
  virtual Ptr clone() const;

  struct DrawContext {
    M::motr::TagLibrary *tagLibrary = nullptr;
    ImVec2 offset{};
    ImDrawList *draw_list = nullptr;
    int depth = 0;
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
  };

  virtual void traverse(DrawContext &context);
  virtual void draw(DrawContext &context);

  void setContextPosition(DrawContext &context, bool updateImGuiCursor);

  LayoutNode(std::shared_ptr<LayoutNode> parent);

  template <typename T>
  T *as();

  template <typename T>
  const T *as() const;
};

struct WindowLayoutNode : public LayoutNode {
  using Ptr = std::shared_ptr<WindowLayoutNode>;
  WindowLayoutNode(std::shared_ptr<LayoutNode> parent);
  void draw(DrawContext &context) override;
  static Ptr wrap(std::shared_ptr<LayoutNode> node);
  static Ptr wrapJsonStringView(std::string_view json);
  LayoutNode::Ptr clone() const override;
};

template <typename T>
const LayoutAttributes &getAttributes(const T *node) {
  const LayoutNode *base = reinterpret_cast<const LayoutNode *>(node);
  return base->attrs;
}

struct TextLayoutSchema {
  SCHEMA_DECLARE_NAMED_ATTRIBUTE(
      horizontalAlign,                   // name
      Attribute::HorizontalAlign,        // type
      Attribute::HorizontalAlign::Left); // default value
  SCHEMA_DECLARE_NAMED_ATTRIBUTE(
      verticalAlign,                            // name
      Attribute::VerticalAlign,                 // type
      Attribute::VerticalAlign::Top);           // default value
  SCHEMA_DECLARE_NAMED_ATTRIBUTE(color,         // name
                                 Color::RGBA32, // type
                                 (Color::RGBA32{255, 255, 255,
                                                255})); // default value
};
struct TextLayoutNode : public LayoutNode, public TextLayoutSchema {
  using Ptr = std::shared_ptr<TextLayoutNode>;
  TextLayoutNode(std::shared_ptr<LayoutNode> parent);
  void draw(DrawContext &context) override;
  LayoutNode::Ptr clone() const override;
};

struct ButtonLayoutSchema {
  SCHEMA_DECLARE_NAMED_ATTRIBUTE(color,         // name
                                 Color::RGBA32, // type
                                 (Color::RGBA32{255, 255, 255,
                                                255})); // default value
};

struct ButtonLayoutNode : public LayoutNode, public ButtonLayoutSchema {
  using Ptr = std::shared_ptr<ButtonLayoutNode>;
  ButtonLayoutNode(std::shared_ptr<LayoutNode> parent);
  void draw(DrawContext &context) override;
  LayoutNode::Ptr clone() const override;
};

// Forward declaration for RefLayoutNode - implementation moved to
// RefLayoutNode.h
struct RefLayoutNode;

// Forward declaration for PlotLayoutNode - implementation moved to
// PlotLayoutNode.h enum class PlotType and struct PlotLayoutNode are now in
// PlotLayoutNode.h

template <typename T>
T *LayoutNode::as() {
  if constexpr (std::is_same_v<T, LayoutNode>)
    return this;

  if constexpr (std::is_same_v<T, WindowLayoutNode>)
    if (type == LayoutNodeType::Window)
      return reinterpret_cast<T *>(this);

  if constexpr (std::is_same_v<T, ButtonLayoutNode>)
    if (type == LayoutNodeType::Button)
      return reinterpret_cast<T *>(this);

  if constexpr (std::is_same_v<T, TextLayoutNode>)
    if (type == LayoutNodeType::Text)
      return reinterpret_cast<T *>(this);

  if constexpr (std::is_same_v<T, RefLayoutNode>)
    if (type == LayoutNodeType::Ref)
      return reinterpret_cast<T *>(this);

  if constexpr (std::is_same_v<T, PlotLayoutNode>)
    if (type == LayoutNodeType::Plot)
      return reinterpret_cast<T *>(this);

  return nullptr;
}

template <typename T>
const T *LayoutNode::as() const {
  if constexpr (std::is_same_v<T, LayoutNode>)
    return this;

  if constexpr (std::is_same_v<T, WindowLayoutNode>)
    if (type == LayoutNodeType::Window)
      return reinterpret_cast<const T *>(this);

  if constexpr (std::is_same_v<T, ButtonLayoutNode>)
    if (type == LayoutNodeType::Button)
      return reinterpret_cast<const T *>(this);

  if constexpr (std::is_same_v<T, TextLayoutNode>)
    if (type == LayoutNodeType::Text)
      return reinterpret_cast<const T *>(this);

  if constexpr (std::is_same_v<T, RefLayoutNode>)
    if (type == LayoutNodeType::Ref)
      return reinterpret_cast<const T *>(this);

  if constexpr (std::is_same_v<T, PlotLayoutNode>)
    if (type == LayoutNodeType::Plot)
      return reinterpret_cast<const T *>(this);

  return nullptr;
}

} // namespace M::motr::Gui

#endif // M_MOTR_GUI_LAYOUT_WINDOW_H
