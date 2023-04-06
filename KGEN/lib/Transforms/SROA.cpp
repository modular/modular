//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"

#include "KGEN/KGENDialect/KGENOps.h"
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
  void run(SmallVectorImpl<Operation *> &toDelete) {
    Derived *derived = getDerived();

    // We check if we can perform the optimization first.
    if (derived->canRun()) {
      // Create a new allocation for each scalar in the container.
      derived->createScalarAllocs();

      // For each user of the allocation replace it with the scalar equivilent.
      for (Operation *user : alloc->getUsers()) {
        builder.setInsertionPointAfter(user);
        derived->replaceUser(user, toDelete);
        toDelete.push_back(user);
      }
      toDelete.push_back(alloc);
    }
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
        }
      }
    } else if (auto store = dyn_cast<StoreOp>(user)) {
      auto operand = store.getArg();
      int64_t index = 0;

      // Decompose the store into a store into each alloca.
      for (Value newAlloc : newAllocas) {
        // Extract the sub element from the value we were about to store. Each
        // derived has its own way of extracting an element.
        Value extract = derived->createExtract(store.getLoc(), operand,
                                               builder.getIndexAttr(index));

        // Store that into the subelement instead.
        builder.create<StoreOp>(store.getLoc(), extract, newAlloc);
        index++;
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
    builder.setInsertionPointAfter(alloc);
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
  Value createExtract(mlir::Location loc, Value operand, IntegerAttr index) {
    return builder.create<POP::StructExtractOp>(loc, operand, index);
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

      // We allow loads if they are only then used in GEPs or Gets.
      if (auto load = dyn_cast<POP::LoadOp>(user)) {
        for (Operation *loadUser : load->getUsers()) {
          if (!isa<POP::ArrayGEPOp, POP::ArrayGetOp>(loadUser))
            return false;

          // Allow GEPs through only if the index is constant.
          if (auto gep = dyn_cast<POP::ArrayGEPOp>(loadUser)) {
            APInt index;
            if (!matchPattern(gep.getIndex(), mlir::m_ConstantInt(&index)))
              return false;
          }
        }
      }

      // We only support array GEPs of constant array indexing.
      if (auto gep = dyn_cast<POP::ArrayGEPOp>(user)) {
        APInt index;
        if (!matchPattern(gep.getIndex(), mlir::m_ConstantInt(&index)))
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
    builder.setInsertionPointAfter(alloc);

    Type elem = containerTy.getResolvedElementType();
    auto asPtr = PointerType::get(elem);

    for (int64_t i = 0; i < numElems; ++i) {
      Value v = builder.create<StackAllocationOp>(alloc.getLoc(), asPtr, 1);
      newAllocas.push_back(v);
    }
  }

  /// Create the array specific element extractor op.
  Value createExtract(mlir::Location loc, Value operand, IntegerAttr index) {
    return builder.create<POP::ArrayGetOp>(loc, operand, index);
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

} // namespace

void SROAPass::runOnOperation() {
  OpBuilder builder{getOperation()->getContext()};

  SmallVector<Operation *, 32> toDelete;

  getOperation()->walk([&](StackAllocationOp alloc) {
    // Skip non singleton stack allocations.
    auto count = dyn_cast<IntegerAttr>(alloc.getCount());
    if (!count || count.getInt() != 1)
      return;

    // Stack allocation is always a pointer to something.
    auto ptrType = cast<POP::PointerType>(alloc.getResult().getType());

    // Obviously skip if it we are not dealing with a struct or array.
    if (auto structTy =
            dyn_cast<POP::StructType>(ptrType.getResolvedElementType())) {
      ReplaceStructs replacer{builder, alloc, structTy};
      replacer.run(toDelete);
    } else if (auto arrayTy =
                   dyn_cast<POP::ArrayType>(ptrType.getResolvedElementType())) {
      ReplaceArray replacer{builder, alloc, arrayTy};
      replacer.run(toDelete);
    }
  });

  // Delete the ops which are no longer used.
  for (Operation *op : toDelete)
    op->erase();
}
