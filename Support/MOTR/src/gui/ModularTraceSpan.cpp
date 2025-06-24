//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "motr/JSON.h" // Please include first to hook up the error handler

#include "ModularTraceSpan.h"
#include "motr/Time.h"
#include <limits>
#include <sstream>

using namespace M::motr::Gui;

ModularTraceSpan::ModularTraceSpan(const nlohmann::json &j)
    : pid(j["pid"].get<int64_t>()), tid(j["tid"].get<int64_t>()),
      name(j["name"].get<std::string>()), start(j["ts"].get<int64_t>()),
      dur(j["dur"].get<int64_t>()), parent(nullptr) {
  if (j.contains("args") && j["args"].contains("detail"))
    detail = j["args"]["detail"].get<std::string>();
}

Timestamp ModularTraceSpan::end() const { return start + dur; }

bool ModularTraceSpan::contains(const ModularTraceSpan *other) const {
  return start <= other->start && end() >= other->end();
}

bool ModularTraceSpan::overlaps(const ModularTraceSpan *other) const {
  return !(end() <= other->start || other->end() <= start);
}

std::string ModularTraceSpan::toString(int max_depth, int max_nodes,
                                       int indent) const {
  std::ostringstream oss;
  int node_count = 0;
  toStringHelper(oss, 0, node_count, max_depth, max_nodes, indent);
  return oss.str();
}

void ModularTraceSpan::toStringHelper(std::ostringstream &oss,
                                      int current_depth, int &node_count,
                                      int max_depth, int max_nodes,
                                      int indent) const {
  if ((max_depth > 0 && current_depth > max_depth) ||
      (max_nodes > 0 && node_count >= max_nodes))
    return;

  oss << std::string(current_depth * indent, ' ') << name << " ("
      << dur.microseconds() << "us)";
  if (!detail.empty())
    oss << " *";
  oss << "\n";

  node_count++;

  for (const auto *child : children) {
    child->toStringHelper(oss, current_depth + 1, node_count, max_depth,
                          max_nodes, indent);
  }
}

uint32_t ModularTraceSpan::numDescendants() const {
  uint32_t count = 0;
  for (const auto *child : children) {
    count++;
    count += child->numDescendants();
  }
  return count;
}

namespace M::motr::Gui {
std::vector<ModularTraceSpan *>
resolveTraceHierarchy(std::vector<ModularTraceSpan> &spans) {
  std::vector<ModularTraceSpan *> roots;

  for (auto &span : spans) {
    bool hasParent = false;
    ModularTraceSpan *bestParent = nullptr;
    Duration bestParentDuration = Duration::max();

    for (auto &potentialParent : spans) {
      if (&span != &potentialParent && span.tid == potentialParent.tid &&
          potentialParent.contains(&span) &&
          potentialParent.dur < bestParentDuration) {
        bestParent = &potentialParent;
        bestParentDuration = potentialParent.dur;
        hasParent = true;
      }
    }

    if (bestParent) {
      span.parent = bestParent;
      bestParent->children.push_back(&span);
    }

    if (!hasParent)
      roots.push_back(&span);
  }

  return roots;
}
} // namespace M::motr::Gui

uint32_t ModularTraceSpan::numAncestors() const {
  uint32_t count = 0;
  for (const auto *parent = this->parent; parent != nullptr;
       parent = parent->parent) {
    count++;
  }
  return count;
}
