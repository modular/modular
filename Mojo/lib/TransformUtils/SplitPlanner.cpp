//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Mojo/TransformUtils/SplitPlanner.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <optional>

using namespace M;

#define DEBUG_TYPE "split-planner"

/// The users of each symbol, in index order.
static std::vector<SmallVector<unsigned, 4>>
collectUsers(ArrayRef<SplitSymbol> symbols) {
  std::vector<SmallVector<unsigned, 4>> users(symbols.size());
  for (unsigned idx = 0, e = symbols.size(); idx != e; ++idx)
    for (unsigned dep : symbols[idx].dependencies)
      users[dep].push_back(idx);
  return users;
}

/// The strongly connected component of each symbol, and the number of
/// components, using Tarjan's algorithm over the dependency edges.
static std::vector<unsigned> collectComponents(ArrayRef<SplitSymbol> symbols,
                                               unsigned &numComponents) {
  constexpr unsigned kUnvisited = ~0u;
  unsigned numSymbols = symbols.size();
  std::vector<unsigned> index(numSymbols, kUnvisited);
  std::vector<unsigned> lowlink(numSymbols, kUnvisited);
  std::vector<unsigned> component(numSymbols, kUnvisited);
  llvm::BitVector onStack(numSymbols);
  SmallVector<unsigned> componentStack;
  struct Frame {
    unsigned symbol;
    unsigned nextDependency;
  };
  SmallVector<Frame> frames;
  unsigned nextIndex = 0;
  numComponents = 0;

  for (unsigned root = 0; root != numSymbols; ++root) {
    if (index[root] != kUnvisited)
      continue;
    index[root] = lowlink[root] = nextIndex++;
    componentStack.push_back(root);
    onStack.set(root);
    frames.push_back({root, 0});
    while (!frames.empty()) {
      unsigned symbol = frames.back().symbol;
      ArrayRef<unsigned> dependencies = symbols[symbol].dependencies;
      if (frames.back().nextDependency < dependencies.size()) {
        unsigned dependency = dependencies[frames.back().nextDependency++];
        if (index[dependency] == kUnvisited) {
          index[dependency] = lowlink[dependency] = nextIndex++;
          componentStack.push_back(dependency);
          onStack.set(dependency);
          frames.push_back({dependency, 0});
        } else if (onStack.test(dependency)) {
          lowlink[symbol] = std::min(lowlink[symbol], index[dependency]);
        }
        continue;
      }

      // `symbol` is finished, so its component is complete when it is the
      // root of that component.
      if (lowlink[symbol] == index[symbol]) {
        while (true) {
          unsigned member = componentStack.pop_back_val();
          onStack.reset(member);
          component[member] = numComponents;
          if (member == symbol)
            break;
        }
        ++numComponents;
      }
      frames.pop_back();
      if (!frames.empty())
        lowlink[frames.back().symbol] =
            std::min(lowlink[frames.back().symbol], lowlink[symbol]);
    }
  }
  return component;
}

/// The symbols reachable from `root`. The walk stops at the symbols in
/// `stopAt` other than `root`, which keeps one outlined subtree out of
/// another.
static SmallVector<unsigned> collectReachable(ArrayRef<SplitSymbol> symbols,
                                              unsigned root,
                                              const llvm::BitVector *stopAt) {
  SmallVector<unsigned> members;
  DenseSet<unsigned> seen;
  SmallVector<unsigned> worklist{root};
  while (!worklist.empty()) {
    unsigned idx = worklist.pop_back_val();
    if (idx != root && stopAt && stopAt->test(idx))
      continue;
    if (!seen.insert(idx).second)
      continue;
    members.push_back(idx);
    llvm::append_range(worklist, symbols[idx].dependencies);
  }
  return members;
}

/// Longest-processing-time makespan. Within 4/3 of optimal, which is NP-hard,
/// and well inside the error of using instruction counts for time.
static uint64_t estimateMakespan(ArrayRef<uint64_t> sizes, unsigned threads) {
  SmallVector<uint64_t> load(threads, 0);
  SmallVector<uint64_t> sorted(sizes);
  llvm::sort(sorted, std::greater<uint64_t>());
  for (uint64_t size : sorted)
    *llvm::min_element(load) += size;
  return *llvm::max_element(load);
}

/// Picks the functions worth a slice of their own.
///
/// A slice is the transitive closure of one anchor, so a function that
/// several anchors reach is cloned into each of their slices and optimized
/// again in each. Outlining it emits one slice instead and leaves a
/// declaration in the slices that referenced it:
///
///     before                          after
///     A -- R -- S, T                  A -- declare R
///     B -- R -- S, T                  B -- declare R
///     (R, S, T optimized twice)       R -- S, T  (optimized once)
///
/// In four steps:
///   1. Walk each anchor's closure, recording which anchors reach each symbol
///      and how many instructions each slice currently carries.
///   2. For a function two or more anchors reach, take the part of its
///      subtree it dominates and score it. That IR stops being optimized in
///      all but one slice, less the shared tail copied into the new slice.
///      A non-positive score means cloning is cheaper, so it is dropped.
///   3. Sort by score, ties broken by name.
///   4. Accept greedily while the estimated critical path over `threads`
///      does not grow, updating the slice sizes after each acceptance.
static llvm::BitVector
pickOutlineRoots(ArrayRef<SplitSymbol> symbols,
                 ArrayRef<SmallVector<unsigned, 4>> users, unsigned threads) {
  constexpr unsigned kNone = ~0u;
  unsigned numSymbols = symbols.size();

  // Which anchors reach each symbol, as positions in `sliceSize`, and how big
  // each anchor's closure is.
  std::vector<SmallVector<unsigned, 2>> reachedBy(numSymbols);
  SmallVector<uint64_t> sliceSize;
  std::vector<unsigned> seenBy(numSymbols, kNone);
  SmallVector<unsigned> worklist;
  for (unsigned anchor = 0; anchor != numSymbols; ++anchor) {
    if (!symbols[anchor].isAnchor)
      continue;
    unsigned position = sliceSize.size();
    uint64_t size = 0;
    worklist.assign(1, anchor);
    while (!worklist.empty()) {
      unsigned idx = worklist.pop_back_val();
      if (seenBy[idx] == position)
        continue;
      seenBy[idx] = position;
      size += symbols[idx].instructionCount;
      if (idx != anchor)
        reachedBy[idx].push_back(position);
      llvm::append_range(worklist, symbols[idx].dependencies);
    }
    sliceSize.push_back(size);
  }

  struct Candidate {
    unsigned root;
    /// Everything the new slice holds.
    uint64_t size;
    /// The part that leaves the slices that used to clone it.
    uint64_t exclusive;
    uint64_t saving;
  };
  SmallVector<Candidate> candidates;
  llvm::BitVector dominated(numSymbols);
  for (unsigned root = 0; root != numSymbols; ++root) {
    if (reachedBy[root].size() < 2 || !symbols[root].canOutline)
      continue;

    // Only the dominated part leaves the other slices. The shared tail stays
    // there and gains a copy here. Requiring the whole closure to be
    // dominated rejects every candidate worth taking.
    SmallVector<unsigned> members = collectReachable(symbols, root, nullptr);
    dominated.set(root);
    bool changed = true;
    while (changed) {
      changed = false;
      for (unsigned member : members) {
        if (dominated.test(member) || users[member].empty())
          continue;
        if (llvm::all_of(users[member],
                         [&](unsigned user) { return dominated.test(user); })) {
          dominated.set(member);
          changed = true;
        }
      }
    }

    uint64_t size = 0, exclusive = 0;
    for (unsigned member : members) {
      size += symbols[member].instructionCount;
      if (dominated.test(member))
        exclusive += symbols[member].instructionCount;
      dominated.reset(member);
    }
    // A non-positive saving means cloning is cheaper.
    uint64_t shared = size - exclusive;
    uint64_t gained = exclusive * (reachedBy[root].size() - 1);
    if (gained <= shared)
      continue;
    candidates.push_back({root, size, exclusive, gained - shared});
  }

  // The best saving comes first. Ties go by name and then by index, so that a
  // build is reproducible.
  llvm::sort(candidates, [&](const Candidate &lhs, const Candidate &rhs) {
    if (lhs.saving != rhs.saving)
      return lhs.saving > rhs.saving;
    StringRef lhsName = symbols[lhs.root].name;
    StringRef rhsName = symbols[rhs.root].name;
    if (lhsName != rhsName)
      return lhsName < rhsName;
    return lhs.root < rhs.root;
  });

  // Accept while the critical path does not grow. A neutral candidate still
  // wins, since every candidate has a positive saving. A module with nothing
  // worth sharing yields no candidate, so no size threshold is needed.
  llvm::BitVector isOutlineRoot(numSymbols);
  uint64_t makespan = estimateMakespan(sliceSize, threads);
  llvm::BitVector covered(numSymbols);
  SmallVector<uint64_t> trial;
  for (const Candidate &candidate : candidates) {
    if (covered.test(candidate.root))
      continue;

    trial.assign(sliceSize.begin(), sliceSize.end());
    for (unsigned position : reachedBy[candidate.root])
      trial[position] -= std::min(trial[position], candidate.exclusive);
    trial.push_back(candidate.size);
    uint64_t trialMakespan = estimateMakespan(trial, threads);
    if (trialMakespan > makespan)
      continue;

    isOutlineRoot.set(candidate.root);
    makespan = trialMakespan;
    std::swap(sliceSize, trial);
    for (unsigned member : collectReachable(symbols, candidate.root, nullptr))
      if (member != candidate.root)
        covered.set(member);
  }

  LLVM_DEBUG(llvm::dbgs() << "outlining " << isOutlineRoot.count()
                          << " subtrees, critical path " << makespan
                          << " insts over " << threads << " threads\n");
  return isOutlineRoot;
}

SplitPlan M::planSplitPerExported(ArrayRef<SplitSymbol> symbols,
                                  unsigned outlineThreads) {
  constexpr unsigned kNone = ~0u;
  unsigned numSymbols = symbols.size();
  SplitPlan plan;
  if (llvm::count_if(symbols, [](const SplitSymbol &symbol) {
        return symbol.isAnchor;
      }) < 2)
    return plan;

  llvm::BitVector isOutlineRoot(numSymbols);
  if (outlineThreads != 0)
    isOutlineRoot = pickOutlineRoots(symbols, collectUsers(symbols),
                                     std::max(2u, outlineThreads));

  // The closure of one anchor, and the group that it merges into when it
  // shares an anchor or a symbol that ties its users with another closure.
  struct Closure {
    unsigned root;
    SmallVector<unsigned> members;
    std::optional<unsigned> group;
  };
  std::vector<Closure> closures;
  std::vector<unsigned> closureOf(numSymbols, kNone);
  for (unsigned idx = 0; idx != numSymbols; ++idx) {
    if (!symbols[idx].isAnchor)
      continue;
    closureOf[idx] = static_cast<unsigned>(closures.size());
    closures.push_back({idx, {}, std::nullopt});
  }

  // The closures that must share a slice, keyed by the symbol that ties them.
  llvm::MapVector<unsigned, llvm::SetVector<unsigned>> tiedClosures;
  std::vector<unsigned> visitedBy(numSymbols, kNone);
  SmallVector<unsigned> worklist;
  for (unsigned closureIdx = 0, e = closures.size(); closureIdx != e;
       ++closureIdx) {
    Closure &closure = closures[closureIdx];
    worklist.assign(1, closure.root);
    while (!worklist.empty()) {
      unsigned idx = worklist.pop_back_val();
      // An outline root has a slice of its own. It does not tie the closures
      // that reach it, which would undo the parallelism the split creates.
      if (isOutlineRoot.test(idx) || visitedBy[idx] == closureIdx)
        continue;
      visitedBy[idx] = closureIdx;
      closure.members.push_back(idx);

      // Another anchor is not cloned because its symbol must stay unique.
      // Its own closure covers what it references.
      if (idx != closure.root && closureOf[idx] != kNone) {
        llvm::SetVector<unsigned> &tied = tiedClosures[idx];
        tied.insert(closureIdx);
        tied.insert(closureOf[idx]);
        continue;
      }

      if (symbols[idx].tiesUsers)
        tiedClosures[idx].insert(closureIdx);
      llvm::append_range(worklist, symbols[idx].dependencies);
    }
  }

  std::vector<SmallVector<unsigned>> groups(tiedClosures.size());
  unsigned groupIdx = 0;
  for (auto it = tiedClosures.begin(); it != tiedClosures.end();
       ++it, ++groupIdx) {
    for (unsigned closureIdx : it->second) {
      std::optional<unsigned> previous = closures[closureIdx].group;
      if (!previous || *previous == groupIdx) {
        groups[groupIdx].push_back(closureIdx);
        closures[closureIdx].group = groupIdx;
        continue;
      }
      for (unsigned moved : groups[*previous]) {
        groups[groupIdx].push_back(moved);
        closures[moved].group = groupIdx;
      }
      groups[*previous].clear();
    }
  }

  std::vector<std::vector<unsigned>> &slices = plan.slices;
  std::fill(visitedBy.begin(), visitedBy.end(), kNone);
  for (SmallVector<unsigned> &group : groups) {
    if (group.empty())
      continue;
    unsigned sliceIdx = static_cast<unsigned>(slices.size());
    std::vector<unsigned> &slice = slices.emplace_back();
    for (unsigned closureIdx : group) {
      for (unsigned member : closures[closureIdx].members) {
        if (visitedBy[member] == sliceIdx)
          continue;
        visitedBy[member] = sliceIdx;
        slice.push_back(member);
      }
    }
  }
  for (Closure &closure : closures)
    if (!closure.group)
      slices.emplace_back(closure.members.begin(), closure.members.end());
  for (unsigned idx : isOutlineRoot.set_bits()) {
    plan.outlineRoots.push_back(idx);
    SmallVector<unsigned> members =
        collectReachable(symbols, idx, &isOutlineRoot);
    slices.emplace_back(members.begin(), members.end());
  }

  // Index order within a slice, and the larger slices first so that they
  // start first. Both orders are fixed, which keeps the output reproducible.
  for (std::vector<unsigned> &slice : slices)
    llvm::sort(slice);
  llvm::sort(slices, [](const auto &lhs, const auto &rhs) {
    if (lhs.size() != rhs.size())
      return lhs.size() > rhs.size();
    return lhs < rhs;
  });
  return plan;
}

SplitPlan M::planSplitPerFunction(ArrayRef<SplitSymbol> symbols) {
  unsigned numSymbols = symbols.size();
  std::vector<SmallVector<unsigned, 4>> users = collectUsers(symbols);

  // `deps` collects what the slice of each symbol holds. A symbol that ties
  // its users cannot be split, and neither can a symbol that reaches one.
  // Such a symbol holds its users and what its dependencies hold. A function
  // that can be split stays out of its users' slices, which declare it.
  llvm::BitVector canBeSplit(numSymbols, true);
  std::vector<llvm::BitVector> deps(numSymbols, llvm::BitVector(numSymbols));
  SmallVector<unsigned> worklist;
  for (unsigned idx = 0; idx != numSymbols; ++idx) {
    const SplitSymbol &symbol = symbols[idx];
    if (symbol.tiesUsers)
      canBeSplit.reset(idx);
    if (symbol.isFunction && symbol.isDeclaration)
      continue;
    deps[idx].set(idx);
    worklist.push_back(idx);
  }

  auto unite = [&](unsigned to, unsigned from) {
    if (to == from || deps[to] == deps[from])
      return false;
    deps[to] |= deps[from];
    return true;
  };
  while (!worklist.empty()) {
    unsigned idx = worklist.pop_back_val();
    bool mergeIntoUsers = !symbols[idx].isFunction || !canBeSplit.test(idx);
    for (unsigned user : users[idx]) {
      if (user == idx)
        continue;
      bool changed = mergeIntoUsers && unite(user, idx);
      if (!canBeSplit.test(idx) && canBeSplit.test(user)) {
        canBeSplit.reset(user);
        changed = true;
        for (unsigned userOfUser : users[user])
          deps[user].set(userOfUser);
      }
      if (changed)
        worklist.push_back(user);
    }

    if (canBeSplit.test(idx))
      continue;
    // `unite` grows `deps[idx]`, and the walk visits the bits it adds.
    for (unsigned dep : deps[idx].set_bits())
      unite(idx, dep);
  }

  for (unsigned idx = 0; idx != numSymbols; ++idx)
    if (symbols[idx].tiesUsers)
      for (unsigned user : users[idx])
        deps[idx].set(user);

  // A function with no users but itself roots a slice even when it cannot be
  // split, because no other slice would hold it.
  auto hasOwnSlice = [&](unsigned idx) {
    ArrayRef<unsigned> usersOf = users[idx];
    bool hasOtherUsers =
        !usersOf.empty() && !(usersOf.size() == 1 && usersOf.front() == idx);
    return symbols[idx].isFunction && (canBeSplit.test(idx) || !hasOtherUsers);
  };

  // A component that no symbol outside it reaches has no user whose slice
  // could hold its members. That is already covered for a lone function, but
  // a cycle of functions that cannot be split would otherwise be left out of
  // every slice. Let one of its members root a slice, as a function with no
  // users does.
  unsigned numComponents = 0;
  std::vector<unsigned> component = collectComponents(symbols, numComponents);
  llvm::BitVector reachedFromOutside(numComponents);
  for (unsigned idx = 0; idx != numSymbols; ++idx)
    for (unsigned user : users[idx])
      if (component[user] != component[idx])
        reachedFromOutside.set(component[idx]);
  auto rootsUnreachedComponent = [&](unsigned idx) {
    return symbols[idx].isFunction && !canBeSplit.test(idx) &&
           !reachedFromOutside.test(component[idx]);
  };

  SplitPlan plan;
  llvm::BitVector inSlice(numSymbols);
  for (unsigned root = 0; root != numSymbols; ++root) {
    const SplitSymbol &symbol = symbols[root];
    if (!symbol.isAnchor || inSlice.test(root) ||
        (symbol.isFunction && !hasOwnSlice(root) &&
         !rootsUnreachedComponent(root)))
      continue;
    std::vector<unsigned> &slice = plan.slices.emplace_back();
    for (unsigned dep : deps[root].set_bits()) {
      if (dep != root && hasOwnSlice(dep))
        continue;
      slice.push_back(dep);
      inSlice.set(dep);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "split " << numSymbols << " symbols into "
                          << plan.slices.size() << " slices\n");
  return plan;
}
