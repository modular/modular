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

#ifndef GENERICML_MGPTRANSFORMS_VALUEINDEX_H
#define GENERICML_MGPTRANSFORMS_VALUEINDEX_H

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "Support/AssertStream.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/Support/raw_ostream.h"

namespace M::MGP {

/// A bijection from all Values, OpOperands or Operations in an MGP::ModelOp
/// to unique indexes. Supports using a BitVector to represent sets of such,
/// and an EquivalenceRelation over indexes to represent equivalence relations
/// over such.
///
/// Also supports pretty-printing values/operands/operations, where SSA names
/// are globally disambiguated by a block index. Blocks indexes are assigned
/// by operation order and pre-order.
template <typename V>
class IndexSet {
public:
  ModelOp model;

  /// Null V at index 0, distinct from every V in the model. Encodes
  /// predicates as relations: (x, {}) is false, (x, {distinguished}) true.
  V distinguished = nullptr;

  /// Returns the number of Vs in model, plus one for the distinguished
  /// value.
  size_t size() const { return allVs.size(); }

  bool contains(V v) const { return vIndexMap.contains(v); }

  /// Returns the unique index of V in model, or the distinguished value.
  size_t getIndex(V v) const {
    auto itr = vIndexMap.find(v);
    assert(itr != vIndexMap.end());
    return itr->second;
  }

  /// Returns the V from model for unique index, or the distinguished value.
  V getV(size_t index) const {
    assert(index < allVs.size());
    return allVs[index];
  }

  /// Returns all the Vs in model. See also getAlLCanonicalVs below.
  ArrayRef<V> getAllVs() const { return allVs; }

  //===--------------------------------------------------------------------===//
  // Pretty-printing
  //===--------------------------------------------------------------------===//

  /// Prints a V. For testing only.
  void printV(llvm::raw_ostream &os, V v) { printVImpl(*this, os, v); }

  /// Prints Vs. For testing only.
  void printVs(llvm::raw_ostream &os, ArrayRef<V> vs) {
    os << "{";
    llvm::interleaveComma(vs, os, [&](V v) { printV(os, v); });
    os << "}";
  }

  /// Prints the model in raw form so that the raw SSA names will match up
  /// with those used by printV.
  void printModel(llvm::raw_ostream &os) {
    model.print(os, mlir::OpPrintingFlags().printGenericOpForm());
    os << "\n\n";
  }

  //===--------------------------------------------------------------------===//
  // Representing sets of Vs using bitvectors.
  //===--------------------------------------------------------------------===//

  using Set = llvm::SparseBitVector<>;

  void addToSet(Set &set, V v) const { set.set(getIndex(v)); }

  void removeFromSet(Set &set, V v) const { set.reset(getIndex(v)); }

  bool setContains(const Set &set, V v) const { return set.test(getIndex(v)); }

  Set emptySet() const { return Set(); }

  /// Returns the set containing all Vs.
  Set fullSet() const {
    auto result = emptySet();
    for (unsigned idx : llvm::seq(allVs.size()))
      result.set(idx);
    return result;
  }

  /// Return set containing only v.
  Set unitSet(V v) const {
    Set set;
    set.set(getIndex(v));
    return set;
  }

  /// Returns the Vs in set, in index order.
  auto values(const Set &set) const {
    SmallVector<V> result;
    for (auto idx : set)
      result.push_back(allVs[idx]);
    return result;
  }

  SmallVector<V> setToVector(const Set &set) const { return values(set); }

  //===--------------------------------------------------------------------===//
  // Representing equivalence relations over Vs using indexes.
  //===--------------------------------------------------------------------===//

  using Equivalence = llvm::EquivalenceClasses<size_t>;

  /// Returns the identity equivalence over all Vs in model, plus the
  /// distinguished value.
  Equivalence identityEquivalence() const {
    Equivalence result;
    for (size_t i = 0; i < allVs.size(); ++i)
      result.insert(i);
    return result;
  }

  bool areEquivalent(const Equivalence &equivalence, V lhs, V rhs) const {
    return equivalence.isEquivalent(getIndex(lhs), getIndex(rhs));
  }

  /// Returns the canonical representation for V w.r.t. equivalence for use
  /// during analysis.
  ///
  /// CAUTION: Not stable, use getStableCanonical when printing for unit tests.
  V getCanonical(const Equivalence &equivalence, V v) const {
    auto itr = equivalence.findLeader(getIndex(v));
    assert(itr != equivalence.member_end());
    return getV(*itr);
  }

  /// Returns the lowest-index V in v's equivalence class, for stable unit
  /// test printing.
  ///
  /// CAUTION: Slow, use getCanonical during analysis.
  V getStableCanonical(const Equivalence &equivalence, V v) const {
    auto itr = equivalence.findLeader(getIndex(v));
    assert(itr != equivalence.member_end());
    size_t index = *itr;
    ++itr;
    while (itr != equivalence.member_end()) {
      index = std::min(index, *itr);
      ++itr;
    }
    return getV(index);
  }

  /// Update equivalence to reflect that lhs and rhs are equivalent Vs.
  void setEquivalent(Equivalence &equivalence, V lhs, V rhs) const {
    equivalence.unionSets(getIndex(lhs), getIndex(rhs));
  }

  /// Returns the set of Vs which are equivalent to v under equivalence,
  /// including v itself.
  Set getEquivalenceClass(const Equivalence &equivalence, V v) const {
    Set result = emptySet();
    for (auto itr = equivalence.findLeader(getIndex(v));
         itr != equivalence.member_end(); ++itr) {
      result.set(*itr);
    }
    return result;
  }

  /// Returns the set of all canonical Vs in model under equivalence. See also
  /// getAlLVs above.
  ///
  /// CAUTION: Not stable.
  Set getAllCanonicalVs(const Equivalence &equivalence) const {
    Set result = emptySet();
    for (auto itr = equivalence.begin(), e = equivalence.end(); itr != e;
         ++itr) {
      if ((*itr)->getData() == 0)
        // Ignore distinguished value.
        continue;
      if ((*itr)->isLeader())
        result.set((*itr)->getData());
    }
    return result;
  }

  /// Returns copy of set closed under the given equivalence.
  [[nodiscard]] Set closeSetUnderEquivalence(const Equivalence &equivalence,
                                             const Set &set) const {
    Set result;
    for (auto index : set) {
      for (auto equivItr = equivalence.findLeader(index);
           equivItr != equivalence.member_end(); ++equivItr)
        result.set(*equivItr);
    }
    return result;
  }

  /// Returns copy of set using only canonical Vs w.r.t. equivalence.
  [[nodiscard]] Set toCanonicalForm(const Equivalence &equivalence,
                                    const Set &set) const {
    Set result;
    for (auto index : set)
      result.set(equivalence.getLeaderValue(index));
    return result;
  }

protected:
  IndexSet(ModelOp model) : model(model) { addV(distinguished); }

  void addV(V v) {
    size_t index = vIndexMap.size();
    vIndexMap.insert({v, index});
    allVs.emplace_back(v);
  }

  /// Maps each V to its value index (for bit vector representations).
  DenseMap<V, size_t> vIndexMap;

  /// All the Vs in model in index order.
  SmallVector<V> allVs;

  /// Maps each block to its block index (for printing).
  DenseMap<Block *, size_t> blockIndexMap;

  friend void visitValues(IndexSet<Value> &indexSet, Block *block);
  friend void visitOperands(IndexSet<const OpOperand *> &indexSet,
                            Block *block);
  friend void visitOperations(IndexSet<Operation *> &indexSet, Block *block);

  friend void printVImpl(IndexSet<Value> &indexSet, llvm::raw_ostream &os,
                         Value value);
  friend void printVImpl(IndexSet<const OpOperand *> &indexSet,
                         llvm::raw_ostream &os, const OpOperand *value);
  friend void printVImpl(IndexSet<Operation *> &indexSet, llvm::raw_ostream &os,
                         Operation *op);
  friend IndexSet<Value> createValueIndexSet(ModelOp model);
  friend IndexSet<const OpOperand *> createOperandIndexSet(ModelOp model);
  friend IndexSet<Operation *> createOperationIndexSet(ModelOp model);
};

/// Visits all the values in block and adds them to indexSet.
void visitValues(IndexSet<Value> &indexSet, Block *block);

/// Prints a debug representation of value to os, using indexSet for
/// disambiguation.
///  %x    is %x bound in the 'execute' main block
///  %x[3] is %x bound in the block with index 3
void printVImpl(IndexSet<Value> &indexSet, llvm::raw_ostream &os, Value value);

IndexSet<Value> createValueIndexSet(ModelOp model);

/// Visits all the operands in block and adds them to indexSet.
void visitOperands(IndexSet<const OpOperand *> &indexSet, Block *block);

/// Prints a debug representation of operand to os, using indexSet for
/// disambiguation.
///   %x      is the unique use of %x in the 'execute' main block
///   %x[3,2] is the second use of %x in the block with index 3
void printVImpl(IndexSet<const OpOperand *> &indexSet, llvm::raw_ostream &os,
                const OpOperand *operand);

IndexSet<const OpOperand *> createOperandIndexSet(ModelOp model);

/// Visits all the operations in block and adds them to indexSet.
void visitOperations(IndexSet<Operation *> &indexSet, Block *block);

/// Prints a debug representation of operation to os, using indexSet for
/// disambiguation. We just print the operation's first result, using the
/// same convention as for Values above. Zero-result operations print as [].
void printVImpl(IndexSet<Operation *> &indexSet, llvm::raw_ostream &os,
                Operation *op);

IndexSet<Operation *> createOperationIndexSet(ModelOp model);

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_VALUEINDEX_H
