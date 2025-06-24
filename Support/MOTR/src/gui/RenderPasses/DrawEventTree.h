//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef M_MOTR_GUI_DRAW_EVENT_TREE_H
#define M_MOTR_GUI_DRAW_EVENT_TREE_H

#include "motr/EventTree.h"
#include <unordered_set>
#include <vector>

namespace M::motr::Gui {

struct TreeNodeItem {
  EventTreeNode::Ptr node;
  int depth;
  bool hasChildren;
  bool isExpanded;
  std::vector<EventTreeNode::Ptr> tags; // Store collapsed tag children
};

struct TreeViewState {
  char filter[128] = "";
  int row = 0;
  // Changed logic: expandedNodes now stores COLLAPSED nodes (default is
  // expanded)
  std::unordered_set<EventTreeNode::Ptr> expandedNodes;
  std::vector<TreeNodeItem>
      flattenedNodes;          // Flattened view for virtual scrolling
  int lastTreeGeneration = -1; // Track when to rebuild flattened view
};

// Core tree functions
TreeViewState &treeState();
bool matchesFilterTxt(std::string_view txt);
std::string makeLabel(const EventTreeNode::Ptr &n);
bool nodeOrDescMatches(const EventTreeNode::Ptr &n);

// Tree flattening and rendering
std::vector<TreeNodeItem>
createTreeNodes(const std::vector<EventTreeNode::Ptr> &flatEvents,
                const std::unordered_set<EventTreeNode::Ptr> &collapsedNodes);
void drawEventTreeNodeVirtual(const TreeNodeItem &item);
void drawEventTreeNodes(const std::vector<EventTreeNode::Ptr> &roots);

// Helper functions for age calculation
EventTreeNode::Ptr findParentProcessEvent(const EventTreeNode::Ptr &node);
int64_t calculateAge(const EventTreeNode::Ptr &node);
EventTreeNode::Ptr getPopEvent(const EventTreeNode::Ptr &n);
int64_t getDuration(const EventTreeNode::Ptr &pushEvent);

} // namespace M::motr::Gui

#endif // M_MOTR_GUI_DRAW_EVENT_TREE_H
