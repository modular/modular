//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MODULAR_TRACE_SPAN_H
#define MODULAR_TRACE_SPAN_H

#include "motr/Time.h"
#include "nlohmann/json_fwd.hpp"
#include <sstream>
#include <string>
#include <vector>

namespace M::motr::Gui {
using Duration = M::motr::Time::Duration;
using Timestamp = M::motr::Time::Timestamp;

struct ModularTraceSpan {
  uint64_t pid;
  uint64_t tid;
  std::string name;

  Timestamp start;
  Duration dur;
  std::string detail;

  ModularTraceSpan *parent;
  std::vector<ModularTraceSpan *> children;

  ModularTraceSpan(const nlohmann::json &j);

  bool contains(const ModularTraceSpan *other) const;
  bool overlaps(const ModularTraceSpan *other) const;
  Timestamp end() const;
  std::string toString(int max_depth = 0, int max_nodes = 0,
                       int indent = 2) const;
  uint32_t numDescendants() const;
  uint32_t numAncestors() const;

  void toStringHelper(std::ostringstream &oss, int current_depth,
                      int &node_count, int max_depth, int max_nodes,
                      int indent) const;
};

std::vector<ModularTraceSpan *>
resolveTraceHierarchy(std::vector<ModularTraceSpan> &spans);
} // namespace M::motr::Gui

#endif // MODULAR_TRACE_SPAN_H
