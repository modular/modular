//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "DrawEventTree.h"
#include "GlobalState.h"
#include "imgui.h"
#include "motr/MString.h"
#include "motr/Time.h"
#include <set>

#define FMT_HEADER_ONLY
#include "fmt/format.h"

namespace M::motr::Gui {

TreeViewState &treeState() {
  static TreeViewState s;
  return s;
}

bool matchesFilterTxt(std::string_view txt) {
  auto &st = treeState();
  return *st.filter == 0 || txt.find(st.filter) != std::string_view::npos;
}

std::string makeLabel(const EventTreeNode::Ptr &n) {
  const auto &m = n->message;
  if (m.isaTag()) {
    return MString{m.id, false}.str(true);
  } else {
    return toString(m.type);
  }
}

bool nodeOrDescMatches(const EventTreeNode::Ptr &n) {
  if (!n || n->message.flags == MessageFlags::Pop)
    return false;
  if (*treeState().filter == 0)
    return true;

  bool self = false;
  const auto &m = n->message;
  if (m.flags == MessageFlags::TagStr) {
    std::string_view key = MStringSafe{m.id};
    std::string_view val = MStringSafe{m.ts};
    self = matchesFilterTxt(key) || matchesFilterTxt(val);
  } else if (m.flags == MessageFlags::TagInt) {
    std::string_view key = MStringSafe{m.id};
    std::string val = fmt::format("{}", m.ts);
    self = matchesFilterTxt(key) || matchesFilterTxt(val);
  } else {
    self = matchesFilterTxt(makeLabel(n));
  }

  if (self)
    return true;
  for (auto &c : n->children)
    if (nodeOrDescMatches(c))
      return true;
  return false;
}

EventTreeNode::Ptr findParentProcessEvent(const EventTreeNode::Ptr &node) {
  auto current = node;
  while (current) {
    if (current->message.type == MessageType::Process) {
      return current;
    }
    current = current->getParent();
  }
  return nullptr;
}

int64_t calculateAge(const EventTreeNode::Ptr &node) {
  if (node->message.type == MessageType::Process) {
    // For process events, calculate age relative to program start
    auto &state = globalState();
    return node->message.ts - state.programStart.nanoseconds();
  } else {
    // For other events (including threads), calculate age relative to parent
    // process
    auto processEvent = findParentProcessEvent(node);
    if (processEvent) {
      return node->message.ts - processEvent->message.ts;
    }
  }
  return 0;
}

// Helper function to count non-tag, non-pop children
size_t countNonTagChildren(const EventTreeNode::Ptr &node) {
  size_t count = 0;
  for (auto &c : node->children) {
    if (!c->message.isaTag() && c->message.flags != MessageFlags::Pop) {
      count++;
    }
  }
  return count;
}

// Helper function to calculate depth by traversing parent chain
int calculateDepth(const EventTreeNode::Ptr &node) {
  int depth = 0;
  auto current = node->getParent();
  while (current) {
    // Only count non-tag, non-pop ancestors
    if (!current->message.isaTag() &&
        current->message.flags != MessageFlags::Pop) {
      depth++;
    }
    current = current->getParent();
  }
  return depth;
}

std::vector<TreeNodeItem>
createTreeNodes(const std::vector<EventTreeNode::Ptr> &flatEvents,
                const std::unordered_set<EventTreeNode::Ptr> &collapsedNodes) {
  std::vector<TreeNodeItem> result;

  for (const auto &node : flatEvents) {
    if (!node || node->message.flags == MessageFlags::Pop)
      continue;
    if (!nodeOrDescMatches(node))
      continue;

    // Skip tag events - they will be collapsed into parent nodes
    if (node->message.isaTag())
      continue;

    // Calculate depth by traversing parent chain
    int depth = calculateDepth(node);
    if (depth > 100)
      continue;

    // Only show process spans at the top level (depth 0)
    // But allow all descendants when processes are expanded
    if (depth == 0 && node->message.type != MessageType::Process)
      continue;

    // Check if this node should be visible based on parent expansion state
    bool shouldBeVisible = true;
    auto parent = node->getParent();
    while (parent && shouldBeVisible) {
      if (!parent->message.isaTag() &&
          parent->message.flags != MessageFlags::Pop) {
        // If any ancestor is collapsed, this node should not be visible
        if (collapsedNodes.find(parent) != collapsedNodes.end()) {
          shouldBeVisible = false;
          break;
        }
      }
      parent = parent->getParent();
    }

    if (!shouldBeVisible)
      continue;

    // Separate tags and others
    std::vector<EventTreeNode::Ptr> tags, others;
    for (auto &c : node->children) {
      if (c->message.isaTag())
        tags.push_back(c);
      else if (c->message.flags != MessageFlags::Pop)
        others.push_back(c);
    }

    std::sort(tags.begin(), tags.end(), [](auto &a, auto &b) {
      return MString{a->message.id, false}.str(true) <
             MString{b->message.id, false}.str(true);
    });

    // Only count non-tag, non-pop children for hasChildren
    bool hasChildren = !others.empty();

    // Check if this node is collapsed (collapsedNodes stores collapsed nodes,
    // default is expanded)
    bool isExpanded =
        hasChildren && (collapsedNodes.find(node) == collapsedNodes.end());

    // Add this node to the result list with tags data
    TreeNodeItem item = {node, depth, hasChildren, isExpanded};
    item.tags = tags; // Store tags for display
    result.push_back(item);
  }

  return result;
}

std::pair<std::string_view, std::string_view>
splitDateTime(const std::string_view &isoDateTime) {
  return {isoDateTime.substr(0, 10), isoDateTime.substr(11, 15)};
}

void drawEventTreeNodeVirtual(const TreeNodeItem &item) {
  auto &st = treeState();
  const auto &node = item.node;
  const auto &m = node->message;

  ImGui::TableNextRow();

  // Column 0: Name (with tree node)
  ImGui::TableSetColumnIndex(0);

  // Minimal indentation - 8 pixels per depth level
  constexpr float indentPerLevel = 8.0f;
  ImGui::Indent(item.depth * indentPerLevel);

  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanFullWidth;
  if (!item.hasChildren) {
    flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
  } else {
    flags |= ImGuiTreeNodeFlags_OpenOnArrow;
    if (item.isExpanded) {
      flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }
  }

  // Get the display name similar to DataExporter logic
  TagLibrary::Ptr tagLibrary = node->getTagLibrary();
  bool localOnly = tagLibrary->isLocalOnly();
  tagLibrary->setLocalOnly(false);

  MString nameKey{Constants::name::sv};
  MString traceNameKey{Constants::TraceName::sv};
  MString processIdKey{Constants::ProcessId::sv};
  MString threadIdKey{Constants::ThreadId::sv};
  MString programNameKey{Constants::ProgramName::sv};

  uint64_t processID = tagLibrary->getOptionalU64(processIdKey).value_or(0);
  uint64_t threadID = tagLibrary->getOptionalU64(threadIdKey).value_or(0);

  std::string displayName;
  if (m.type == MessageType::Process) {
    displayName = fmt::format("process {}", processID);
  } else if (m.type == MessageType::Thread) {
    displayName = fmt::format("process {} thread {}", processID, threadID);
  } else {
    std::string traceName{
        tagLibrary->getOptionalString(traceNameKey).value_or("<unknown>")};
    displayName = traceName;
  }

  if (tagLibrary->hasTag(programNameKey)) {
    std::string programName{tagLibrary->getString(programNameKey)};
    displayName = fmt::format("{} {}", programName, displayName);
  }

  tagLibrary->setLocalOnly(localOnly);

  bool nodeClicked = false;
  if (item.hasChildren) {
    bool open = ImGui::TreeNodeEx(node.get(), flags, "%s", displayName.c_str());

    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
      nodeClicked = true;
    }

    if (open != item.isExpanded) {
      if (open) {
        // Node was expanded - remove from collapsed set
        st.expandedNodes.erase(node);
      } else {
        // Node was collapsed - add to collapsed set
        st.expandedNodes.insert(node);
      }
      st.lastTreeGeneration = -1;
    }

    if (open && item.hasChildren) {
      ImGui::TreePop();
    }
  } else {
    ImGui::TreeNodeEx(node.get(), flags, "%s", displayName.c_str());
  }

  ImGui::Unindent(item.depth * indentPerLevel);

  // Column 1: Age (as duration timestring, right-aligned)
  ImGui::TableSetColumnIndex(1);
  int64_t age = calculateAge(node);
  if (age > 0) {
    M::motr::Time::Duration ageDuration(age);
    std::string ageStr =
        ageDuration.toString(M::motr::Time::Precision::Milliseconds);
    float columnWidth = ImGui::GetColumnWidth();
    float textWidth = ImGui::CalcTextSize(ageStr.c_str()).x;
    float offset = columnWidth - textWidth - ImGui::GetStyle().ItemSpacing.x;
    if (offset > 0)
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);
    ImGui::Text("%s", ageStr.c_str());
  } else {
    ImGui::Text("%s", "0s");
  }

  // Column 2: Duration (ns, right-aligned)
  ImGui::TableSetColumnIndex(2);
  int64_t duration = node->getDuration().nanoseconds();

  if (duration == 0) {
    auto &state = globalState();
    duration = state.frameStart.nanoseconds() - int64_t(node->message.ts);
    duration =
        duration - duration % Time::nanosecondsPer<Time::Precision::Seconds>();
  }

  double fduration_ms = double(duration) / 1000000.0;

  std::string durStr = fmt::format("{:12.6f}ms", fduration_ms);
  float columnWidth = ImGui::GetColumnWidth();
  float textWidth = ImGui::CalcTextSize(durStr.c_str()).x;
  float offset = columnWidth - textWidth - ImGui::GetStyle().ItemSpacing.x;
  if (offset > 0)
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);
  ImGui::Text("%s", durStr.c_str());

  // Column 3: Date
  // Column 4: Time
  ImGui::TableSetColumnIndex(3);
  std::string timestamp = M::motr::timeNsToISODate(m.ts);
  auto [dateStr, timeStr] = splitDateTime(timestamp);
  ImGui::TextUnformatted(std::string{dateStr}.c_str());

  ImGui::TableSetColumnIndex(4);
  ImGui::TextUnformatted(std::string{timeStr}.c_str());

  // Column 5: Tags (as buttons)
  ImGui::TableSetColumnIndex(5);
  if (m.flags == MessageFlags::Push) {
    tagLibrary = node->getTagLibrary();
    localOnly = tagLibrary->isLocalOnly();
    tagLibrary->setLocalOnly(true);

    std::unordered_set<Hash::Value> seenKeys;
    seenKeys.insert(nameKey.hash);
    seenKeys.insert(processIdKey.hash);
    seenKeys.insert(threadIdKey.hash);
    seenKeys.insert(traceNameKey.hash);

    MString sourceFileKey{Constants::SourceFile::sv};
    MString sourceLineKey{Constants::SourceLine::sv};
    seenKeys.insert(sourceFileKey.hash);
    seenKeys.insert(sourceLineKey.hash);

    bool first = true;
    for (auto &[keyMstr, value] : tagLibrary->tagStrMap) {
      if (seenKeys.find(keyMstr.hash) != seenKeys.end())
        continue;

      if (!first)
        ImGui::SameLine();
      first = false;

      std::string buttonLabel = fmt::format("{}={}", keyMstr.sv(true), value);

      int imguiId = static_cast<int>(keyMstr.hash.v ^ m.id);
      ImGui::PushID(imguiId);
      if (ImGui::SmallButton(buttonLabel.c_str())) {
        // Set filter to the tag value
        strncpy(st.filter, value.c_str(), sizeof(st.filter) - 1);
        st.filter[sizeof(st.filter) - 1] = '\0';
        st.lastTreeGeneration = -1; // Force rebuild
      }
      ImGui::PopID();
    }

    for (auto &[keyMstr, value] : tagLibrary->tagIntMap) {
      if (seenKeys.find(keyMstr.hash) != seenKeys.end())
        continue;

      if (!first)
        ImGui::SameLine();
      first = false;

      std::string buttonLabel = fmt::format("{}={}", keyMstr.sv(true), value);

      int imguiId = static_cast<int>(keyMstr.hash.v ^ m.id);
      ImGui::PushID(imguiId);
      if (ImGui::SmallButton(buttonLabel.c_str())) {
        // Set filter to the tag value
        std::string valueStr = fmt::format("{}", value);
        strncpy(st.filter, valueStr.c_str(), sizeof(st.filter) - 1);
        st.filter[sizeof(st.filter) - 1] = '\0';
        st.lastTreeGeneration = -1; // Force rebuild
      }
      ImGui::PopID();
    }

    tagLibrary->setLocalOnly(localOnly);
  } else {
    ImGui::TextUnformatted("");
  }

  // Column 6: Source Location (moved to last)
  ImGui::TableSetColumnIndex(6);
  tagLibrary = node->getTagLibrary();
  localOnly = tagLibrary->isLocalOnly();
  tagLibrary->setLocalOnly(false);

  MString sourceFileKey{Constants::SourceFile::sv};
  MString sourceLineKey{Constants::SourceLine::sv};

  std::string sourceLoc{tagLibrary->getString(sourceFileKey)};
  if (!sourceLoc.empty()) {
    uint64_t sourceLine = tagLibrary->getU64(sourceLineKey);
    if (sourceLine > 0) {
      sourceLoc = fmt::format("{}:{}", sourceLoc, sourceLine);
    }
  }
  tagLibrary->setLocalOnly(localOnly);
  ImGui::Text("%s", sourceLoc.c_str());

  st.row++;
}

void drawEventTreeNodes(const std::vector<EventTreeNode::Ptr> &roots) {

  static M::motr::Time::Duration debounceDuration =
      M::motr::Time::Duration::fromSeconds(1);
  auto &st = treeState();

  // Check if we need to rebuild the flattened tree
  int currentGeneration = globalState().generation;
  if (st.lastTreeGeneration != currentGeneration) {
    static M::motr::Time::Timestamp lastTimestamp =
        M::motr::Time::Timestamp::now();
    const M::motr::Time::Timestamp now = M::motr::Time::Timestamp::now();
    const M::motr::Time::Duration duration = now - lastTimestamp;
    if (duration > debounceDuration) {
      // Use the pre-flattened event tree instead of recursive traversal
      const auto &flatEvents = globalState().getFlatEventTree();
      st.flattenedNodes = createTreeNodes(flatEvents, st.expandedNodes);
      st.lastTreeGeneration = currentGeneration;
      lastTimestamp = now;
    }
  }

  // Use ImGuiListClipper for virtual scrolling
  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(st.flattenedNodes.size()));

  while (clipper.Step()) {
    for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
      drawEventTreeNodeVirtual(st.flattenedNodes[i]);
    }
  }
}

} // namespace M::motr::Gui
