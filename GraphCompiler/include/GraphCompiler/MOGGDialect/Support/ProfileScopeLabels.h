//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Location-walking helpers for Graph profile scope NVTX labels, shared by
// `WithTraceOp::emitMojo` and `AddProfileScopeTracing.cpp`.
// `Graph.profile_scope` (max/python/max/graph/graph.py) wraps each op's
// location in a `ProfileScopeLocationAttr`; the innermost scope ends up as the
// outermost layer.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_PROFILESCOPELABELS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_PROFILESCOPELABELS_H

#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "Support/ADT/DenseStringMap.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>
#include <string>

namespace M::MOGG {

/// One profile_scope nesting layer, outermost scope first (e.g.
/// `{"draft_forward", "inner_stage"}`).
using ProfileScopeLayer = llvm::SmallVector<std::string, 2>;

} // namespace M::MOGG

namespace M::MOGG {

/// A contiguous run of ops in a region body that share a profile_scope layer
/// set.
struct ProfileScopeLayerSet {
  mlir::Block::iterator first;
  /// Inclusive.
  mlir::Block::iterator last;
  llvm::SmallVector<ProfileScopeLayer, 2> layers;

  llvm::iterator_range<mlir::Block::iterator> ops() const {
    return llvm::make_range(first, std::next(last));
  }
};

namespace detail {

using ProfileScopeLayerSetVector = llvm::SmallSetVector<ProfileScopeLayer, 2>;

inline void walkProfileScopeLayer(mlir::Location loc,
                                  llvm::ArrayRef<std::string> prefix,
                                  ProfileScopeLayerSetVector &layers);

/// Walks each sub-location as an independent peer with no prefix.
inline void recurseLocPeers(llvm::ArrayRef<mlir::Location> subs,
                            ProfileScopeLayerSetVector &layers) {
  for (mlir::Location sub : subs)
    walkProfileScopeLayer(sub, {}, layers);
}

/// Prepends `prefix` to every layer found below each sub-location, emitting
/// `prefix` alone for a sub-location with no layers since the outer scopes
/// still apply.
inline void foldLocsIntoPrefix(llvm::ArrayRef<mlir::Location> subs,
                               llvm::ArrayRef<std::string> prefix,
                               ProfileScopeLayerSetVector &layers) {
  for (mlir::Location sub : subs) {
    ProfileScopeLayerSetVector subLayers;
    walkProfileScopeLayer(sub, {}, subLayers);
    for (const ProfileScopeLayer &subLayer : subLayers) {
      ProfileScopeLayer combined;
      combined.reserve(prefix.size() + subLayer.size());
      combined.append(prefix.begin(), prefix.end());
      combined.append(subLayer.begin(), subLayer.end());
      layers.insert(std::move(combined));
    }
    if (subLayers.empty())
      layers.insert(ProfileScopeLayer(prefix.begin(), prefix.end()));
  }
}

/// Collects every reachable profile_scope layer. `prefix` holds the labels
/// accumulated from outer wrappers, outermost first, and is emitted at a leaf.
///
/// `FusedLoc` and `CallSiteLoc` are peer boundaries only when `prefix` is
/// empty, since passes (CSE, SequenceFusion, ...) may merge locations from
/// non-nested `profile_scope()` calls. Under a non-empty prefix the node is
/// nested in active scopes (e.g. `AttachPythonStackTraces` turned an
/// `OpaqueLoc` into a `FusedLoc`), so the prefix applies to every sub-layer.
inline void walkProfileScopeLayer(mlir::Location loc,
                                  llvm::ArrayRef<std::string> prefix,
                                  ProfileScopeLayerSetVector &layers) {
  if (auto profileLoc = mlir::dyn_cast<ProfileScopeLocationAttr>(loc)) {
    // This scope encloses those in `prefix`, so prepend it.
    ProfileScopeLayer nextPrefix;
    nextPrefix.reserve(prefix.size() + 1);
    nextPrefix.push_back(profileLoc.getName().str());
    nextPrefix.append(prefix.begin(), prefix.end());
    walkProfileScopeLayer(profileLoc.getChildLoc(), nextPrefix, layers);
    return;
  }
  if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    if (prefix.empty())
      recurseLocPeers(fusedLoc.getLocations(), layers);
    else
      foldLocsIntoPrefix(fusedLoc.getLocations(), prefix, layers);
    return;
  }
  if (auto callSiteLoc = mlir::dyn_cast<mlir::CallSiteLoc>(loc)) {
    if (prefix.empty()) {
      walkProfileScopeLayer(callSiteLoc.getCallee(), prefix, layers);
      walkProfileScopeLayer(callSiteLoc.getCaller(), prefix, layers);
    } else {
      foldLocsIntoPrefix({callSiteLoc.getCallee(), callSiteLoc.getCaller()},
                         prefix, layers);
    }
    return;
  }
  if (!prefix.empty())
    layers.insert(ProfileScopeLayer(prefix.begin(), prefix.end()));
}

inline void collectProfileScopeLayersImpl(mlir::Location loc,
                                          ProfileScopeLayerSetVector &layers) {
  walkProfileScopeLayer(loc, {}, layers);
}

/// Peer layers have no inherent order (it follows a `FusedLoc`'s input order),
/// so they are sorted lexicographically.
inline bool profileScopeLayerLess(const ProfileScopeLayer &a,
                                  const ProfileScopeLayer &b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

inline void
collectProfileScopeColorsImpl(mlir::Location loc,
                              llvm::SmallVectorImpl<std::string> &colors) {
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](ProfileScopeLocationAttr profileLoc) {
    if (auto color = profileLoc.getColor()) {
      if (!llvm::is_contained(colors, color))
        colors.push_back(color.str());
    }
    return mlir::WalkResult::advance();
  });
  walker.walk(loc);
}

/// Joins labels with `/` within a layer and layers with `:`.
inline void joinProfileScopeLayers(llvm::ArrayRef<ProfileScopeLayer> layers,
                                   llvm::raw_ostream &os) {
  llvm::interleave(
      layers,
      [&](const ProfileScopeLayer &layer) {
        llvm::interleave(
            layer, [&](const std::string &s) { os << s; }, [&] { os << '/'; });
      },
      [&] { os << ':'; });
}

} // namespace detail

/// Returns deduplicated layers in canonical order. Multiple layers come from
/// compiler pass merges.
inline llvm::SmallVector<ProfileScopeLayer, 2>
collectProfileScopeLayers(mlir::Location loc) {
  detail::ProfileScopeLayerSetVector layers;
  detail::collectProfileScopeLayersImpl(loc, layers);

  auto out = layers.takeVector();
  llvm::sort(out, detail::profileScopeLayerLess);
  return out;
}

inline llvm::SmallVector<std::string, 2>
collectProfileScopeColors(mlir::Location loc) {
  llvm::SmallVector<std::string, 2> colors;
  detail::collectProfileScopeColorsImpl(loc, colors);
  llvm::sort(colors);
  return colors;
}

/// Conflicting colors (e.g. merged peer scopes requesting different colors)
/// resolve to no color rather than an arbitrary pick.
inline std::optional<std::string>
resolveProfileScopeColor(llvm::ArrayRef<std::string> colors) {
  if (colors.size() == 1)
    return colors.front();
  return std::nullopt;
}

/// Formats `layers` (in `collectProfileScopeLayers` order) as `"[a/b:c]"`,
/// with a trailing space as a prefix or a leading space as a suffix. Returns
/// "" for no layers so callers can prepend unconditionally.
inline std::string formatProfileScope(llvm::ArrayRef<ProfileScopeLayer> layers,
                                      bool asPrefix) {
  if (layers.empty())
    return "";
  std::string result;
  llvm::raw_string_ostream os(result);
  if (!asPrefix)
    os << ' ';
  os << '[';
  detail::joinProfileScopeLayers(layers, os);
  os << ']';
  if (asPrefix)
    os << ' ';
  return result;
}

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_PROFILESCOPELABELS_H
