//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/Analysis/CFG.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace POP;

namespace M::KGEN {
#define GEN_PASS_DEF_SROA
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class SROAPass : public M::KGEN::impl::SROABase<SROAPass> {
public:
  void runOnOperation() override;
};

/// Base helper class using CRTP to wrap most of the common logic between the
/// array and struct replacers.
template <typename Derived, typename ContainerType>
struct Replacer {
  OpBuilder &builder;

  /// The allocation being replaced.
  StackAllocationOp alloc;

  /// The type of the non-scalar we are turning into scalars. I.E ArrayType /
  /// StructType.
  ContainerType containerTy;

  /// The new scalar stack allocations we have created.
  SmallVector<Value> newAllocas;

  Replacer(OpBuilder &builder, StackAllocationOp alloc, ContainerType container)
      : builder(builder), alloc(alloc), containerTy(container) {}

  Derived *getDerived() { return static_cast<Derived *>(this); }

  /// Run the main replacement loop, going through all the uses of the stack and
  /// swapping them out for scalar equivalents.
  bool run(SmallVectorImpl<Operation *> &toDelete) {
    Derived *derived = getDerived();

    // We check if we can perform the optimization first.
    if (!derived->canRun())
      return false;

    // Create a new allocation for each scalar in the container.
    builder.setInsertionPointAfter(alloc);
    derived->createScalarAllocs();

    // For each user of the allocation replace it with the scalar equivilent.
    for (Operation *user : alloc->getUsers()) {
      builder.setInsertionPointAfter(user);
      derived->replaceUser(user, toDelete);
      toDelete.push_back(user);
    }
    toDelete.push_back(alloc);

    return true;
  }

  // Handle the cases which are the same for both array and struct (load/store)
  // then delegate any remaining to the derived class.
  void replaceUser(Operation *user, SmallVectorImpl<Operation *> &toDelete) {
    Derived *derived = getDerived();

    if (auto load = dyn_cast<POP::LoadOp>(user)) {
      // Store each load in its index in the array, using the fact that C++ will
      // make value null by default.
      SmallVector<Value> loadedVals(newAllocas.size());

      // Get the load for the given index in the aggregate or create a load to
      // the equivelent scalar.
      auto getOrCreateLoad = [&](uint64_t index) {
        Value newVal = loadedVals[index];
        if (!newVal) {
          newVal =
              builder.create<POP::LoadOp>(load.getLoc(), newAllocas[index]);
          loadedVals[index] = newVal;
        }
        return newVal;
      };

      // A load from an allocation of N is implicitly a load from the first
      // dimension.
      if (std::is_same<ContainerType, POP::StackAllocationOp>::value) {
        load.replaceAllUsesWith(getOrCreateLoad(0));
        return;
      }

      // Replace the *user* of each load with the loaded scalar or for GEPs the
      // pointer itself.
      for (Operation *loadUser : load->getUsers()) {
        if (auto gep = dyn_cast<POP::StructGEPOp>(loadUser)) {
          gep.replaceAllUsesWith(newAllocas[gep.getIndexAttr().getInt()]);
          toDelete.push_back(gep);
        } else if (auto extract = dyn_cast<POP::StructExtractOp>(loadUser)) {
          Value newVal = getOrCreateLoad(extract.getIndex().getLimitedValue());
          extract.replaceAllUsesWith(newVal);
          toDelete.push_back(extract);
        } else if (auto get = dyn_cast<POP::ArrayGetOp>(loadUser)) {
          auto attr = cast<IntegerAttr>(get.getIndex());
          Value newVal = getOrCreateLoad(attr.getInt());
          get.replaceAllUsesWith(newVal);
          toDelete.push_back(get);
        } else if (auto gep = dyn_cast<POP::ArrayGEPOp>(loadUser)) {
          APInt index;
          matchPattern(gep.getIndex(), mlir::m_ConstantInt(&index));
          gep.replaceAllUsesWith(newAllocas[index.getLimitedValue()]);
          toDelete.push_back(gep);
        }
      }
    } else if (auto store = dyn_cast<StoreOp>(user)) {
      auto operand = store.getArg();
      int64_t index = 0;

      // Decompose the store into a store into each alloca.
      for (Value newAlloc : newAllocas) {
        // Extract the sub element from the value we were about to store. Each
        // derived has its own way of extracting an element.
        Value extract = derived->createExtract(store.getLoc(), operand, index);

        // Store that into the subelement instead.
        builder.create<StoreOp>(store.getLoc(), extract, newAlloc);
        ++index;

        // Stack of >1 stores are implicity only a reference to the first
        // element so we can stop after the first store.
        if constexpr (std::is_same<ContainerType,
                                   POP::StackAllocationOp>::value)
          break;
      }
    } else {
      derived->replaceUserImpl(user, toDelete);
    }
  }
};

/// The extra helper class for structures.
struct ReplaceStructs : public Replacer<ReplaceStructs, POP::StructType> {
  using ContainerType = POP::StructType;

  ReplaceStructs(OpBuilder &builder, StackAllocationOp alloc,
                 ContainerType container)
      : Replacer(builder, alloc, container) {}

  bool canRun() {
    for (Operation *user : alloc->getUsers()) {
      // If the user is something which actually expects the full structure like
      // a call then we cannot perfom the optimization.
      if (!isa<POP::StructGEPOp, POP::StructExtractOp, POP::StoreOp,
               POP::LoadOp>(user))
        return false;

      // If the user is the argument of the store, then we cannot elide.
      if (auto store = dyn_cast<POP::StoreOp>(user))
        if (store.getArg() == alloc)
          return false;

      // We can SROA loads if they are only used in extract ops.
      if (auto load = dyn_cast<POP::LoadOp>(user)) {
        for (Operation *loadUser : load->getUsers()) {
          if (!isa<POP::StructGEPOp, POP::StructExtractOp>(loadUser))
            return false;
        }
      }
    }
    return true;
  }

  // Allocate the scalars which should replace the main alloc.
  void createScalarAllocs() {
    newAllocas.reserve(containerTy.getNumElements());
    for (Type elem : containerTy.getParameterizedElementTypes()) {
      auto asPtr = PointerType::get(elem);
      Value v = builder.create<StackAllocationOp>(alloc.getLoc(), asPtr, 1);
      newAllocas.push_back(v);
    }
  }

  /// Replace some of the struct specific things.
  void replaceUserImpl(Operation *user,
                       SmallVectorImpl<Operation *> &toDelete) {
    if (auto gep = dyn_cast<StructGEPOp>(user))
      gep.replaceAllUsesWith(newAllocas[gep.getIndexAttr().getInt()]);
    else if (auto extract = dyn_cast<StructExtractOp>(user))
      extract.replaceAllUsesWith(newAllocas[extract.getIndexAttr().getInt()]);
  }

  /// The extractor op for structures.
  Value createExtract(mlir::Location loc, Value operand, int64_t index) {
    return builder.create<POP::StructExtractOp>(loc, operand,
                                                builder.getIndexAttr(index));
  }
};

/// The extra helper class for arrays.
struct ReplaceArray : public Replacer<ReplaceArray, POP::ArrayType> {
  using ContainerType = POP::ArrayType;

  ReplaceArray(OpBuilder &builder, StackAllocationOp alloc,
               ContainerType container)
      : Replacer(builder, alloc, container) {}

  bool canRun() {
    // If we don't know the size of the array there's nothing to do.
    if (!containerTy.getResolvedSize())
      return false;

    for (Operation *user : alloc->getUsers()) {
      // If the user is something which actually expects the full structure like
      // a call then we cannot perfom the optimization.
      if (!isa<POP::ArrayGEPOp, POP::StoreOp, POP::LoadOp>(user))
        return false;

      // If the user is the argument of the store, then we cannot elide.
      if (auto store = dyn_cast<POP::StoreOp>(user))
        if (store.getArg() == alloc)
          return false;

      // We allow loads if they are only then used in GEPs or Gets.
      if (auto load = dyn_cast<POP::LoadOp>(user)) {
        for (Operation *loadUser : load->getUsers()) {
          if (!isa<POP::ArrayGEPOp, POP::ArrayGetOp>(loadUser))
            return false;

          // Allow GEPs through only if the index is constant.
          if (auto gep = dyn_cast<POP::ArrayGEPOp>(loadUser)) {
            APInt index;
            if (!matchPattern(gep.getIndex(), mlir::m_ConstantInt(&index)) ||
                index.isNegative())
              return false;

            // Oddly this comes up. Guard against out of range accesses.
            if (static_cast<int64_t>(index.getLimitedValue()) >=
                *containerTy.getResolvedSize())
              return false;
          }
        }
      }

      // We only support array GEPs of constant array indexing.
      if (auto gep = dyn_cast<POP::ArrayGEPOp>(user)) {
        APInt index;
        if (!matchPattern(gep.getIndex(), mlir::m_ConstantInt(&index)) ||
            index.isNegative())
          return false;

        // Oddly this comes up. Guard against out of range accesses.
        if (static_cast<int64_t>(index.getLimitedValue()) >=
            *containerTy.getResolvedSize())
          return false;
      }
    }

    return true;
  }

  /// Allocate the scalar stack allocations which replace the single array
  /// allocation.
  void createScalarAllocs() {
    int64_t numElems = *containerTy.getResolvedSize();
    newAllocas.reserve(numElems);

    Type elem = containerTy.getElementAsType();
    auto asPtr = PointerType::get(elem);
    for (int64_t i = 0; i < numElems; ++i) {
      Value v = builder.create<StackAllocationOp>(alloc.getLoc(), asPtr, 1);
      newAllocas.push_back(v);
    }
  }

  /// Create the array specific element extractor op.
  Value createExtract(mlir::Location loc, Value operand, int64_t index) {
    return builder.create<POP::ArrayGetOp>(loc, operand,
                                           builder.getIndexAttr(index));
  }

  /// Handle the array specific ops.
  void replaceUserImpl(Operation *user,
                       SmallVectorImpl<Operation *> &toDelete) {
    if (auto gep = dyn_cast<ArrayGEPOp>(user)) {
      APInt index;
      matchPattern(gep.getIndex(), mlir::m_ConstantInt(&index));
      gep.replaceAllUsesWith(newAllocas[index.getLimitedValue()]);
    }
  }
};

/// In this case we treat the underlaying stack allocation as the container
/// itself.
struct ReplaceStack : public Replacer<ReplaceStack, POP::StackAllocationOp> {
  using ContainerType = POP::StackAllocationOp;

  ReplaceStack(OpBuilder &builder, StackAllocationOp alloc)
      : Replacer(builder, alloc, alloc) {}

  bool canRun() {
    for (Operation *user : alloc->getUsers()) {
      if (!isa<POP::OffsetOp, POP::StoreOp, POP::LoadOp, POP::ArrayGEPOp,
               POP::StructGEPOp>(user))
        return false;

      // If the user is the argument of the store, then we cannot elide.
      if (auto store = dyn_cast<POP::StoreOp>(user))
        if (store.getArg() == alloc)
          return false;

      // We only support offsets with constant terms.
      if (auto offset = dyn_cast<POP::OffsetOp>(user)) {
        APInt index;
        if (!matchPattern(offset.getIndex(), mlir::m_ConstantInt(&index)) ||
            index.isNegative())
          return false;

        // Oddly this comes up. Guard against out of range accesses.
        if (static_cast<int64_t>(index.getLimitedValue()) >=
            cast<IntegerAttr>(alloc.getCount()).getInt())
          return false;
      }
    }

    return true;
  }

  /// Allocate the scalar stack allocations which replace the single array
  /// allocation.
  void createScalarAllocs() {
    // The allocation is the aggregate in this case.
    int64_t numElems = cast<IntegerAttr>(alloc.getCount()).getInt();
    newAllocas.reserve(numElems);

    auto ptr = cast<POP::PointerType>(alloc.getResult().getType());
    for (int64_t i = 0; i < numElems; ++i) {
      Value v = builder.create<StackAllocationOp>(alloc.getLoc(), ptr, 1);
      newAllocas.push_back(v);
    }
  }

  /// An extraction of something being stored to stack of N is always just the
  /// operand itself.
  Value createExtract(mlir::Location loc, Value operand, int64_t index) {
    return operand;
  }

  void replaceUserImpl(Operation *user,
                       SmallVectorImpl<Operation *> &toDelete) {
    if (auto offset = dyn_cast<OffsetOp>(user)) {
      APInt index;
      matchPattern(offset.getIndex(), mlir::m_ConstantInt(&index));
      offset.replaceAllUsesWith(newAllocas[index.getLimitedValue()]);
    } else if (auto gep = dyn_cast<ArrayGEPOp>(user)) {
      // An array GEP is implicitly on the first element so swap it for a GEP on
      // that allocation. We don't need to check the index because if it is
      // illegal in the new one then it was illegal on the old gep. The index
      // refers to the element in the array, the gep is always on the first
      // element of the stack.
      auto newGep = builder.create<ArrayGEPOp>(gep.getLoc(), newAllocas[0],
                                               gep.getIndex());
      gep.replaceAllUsesWith(newGep.getResult());
    } else if (auto gep = dyn_cast<StructGEPOp>(user)) {
      // Dito with struct geps, index is always legal.
      auto newGep = builder.create<StructGEPOp>(gep.getLoc(), newAllocas[0],
                                                gep.getIndex());
      gep.replaceAllUsesWith(newGep.getResult());
    }
  }
};

} // namespace

void SROAPass::runOnOperation() {
  OpBuilder builder{getOperation()->getContext()};

  // The loop limit is an arbritary value to provide an upperbound on compile
  // time. However from experimentation this pass does not take a significant
  // amount of time to run and is a net-postive on compile time.
  constexpr size_t loopLimit = 10;

  size_t iters = 0;

  bool changed = true;
  while (changed && iters < loopLimit) {
    changed = false;
    iters++;

    SmallVector<Operation *, 32> toDelete;

    getOperation()->walk([&](StackAllocationOp alloc) {
      // Skip non singleton stack allocations.
      auto count = dyn_cast<IntegerAttr>(alloc.getCount());
      if (!count)
        return;

      size_t numElems = count.getInt();

      // We won't try to decompose large stack allocations.
      if (numElems > 16)
        return;

      // Stack allocation is always a pointer to something.
      auto ptrType = cast<POP::PointerType>(alloc.getResult().getType());

      // We decompose structs if there is one element otherwise we decompose the
      // stack itself.
      if (numElems != 1) {
        // Replace stack of N with N stacks of 1.
        ReplaceStack replacer{builder, alloc};
        changed |= replacer.run(toDelete);
      } else if (auto structTy =
                     dyn_cast<POP::StructType>(ptrType.getElementAsType())) {
        ReplaceStructs replacer{builder, alloc, structTy};
        changed |= replacer.run(toDelete);
      } else if (auto arrayTy =
                     dyn_cast<POP::ArrayType>(ptrType.getElementAsType())) {
        ReplaceArray replacer{builder, alloc, arrayTy};
        changed |= replacer.run(toDelete);
      }
    });

    // Delete the ops which are no longer used.
    for (Operation *op : toDelete)
      op->erase();
  }

  // Control-flow is not modified.
  markAnalysesPreserved<HLCF::CFGAnalysis>();
}
