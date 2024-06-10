//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/HLCFDialect/Analysis/CFG.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
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
class SROAPass : public impl::SROABase<SROAPass> {
public:
  using SROABase::SROABase;
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

  /// The maximum number of elements to decompose.
  uint32_t maxNumElements;

  /// The new scalar stack allocations we have created.
  SmallVector<Value> &newAllocas;

  Replacer(OpBuilder &builder, StackAllocationOp alloc, ContainerType container,
           uint32_t maxNumElements, SmallVector<Value> &valueMemCache)
      : builder(builder), alloc(alloc), containerTy(container),
        maxNumElements(maxNumElements), newAllocas(valueMemCache) {}

  ~Replacer() { newAllocas.clear(); }

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
    for (Operation *user : llvm::make_early_inc_range(alloc->getUsers())) {
      builder.setInsertionPointAfter(user);
      derived->replaceUser(user, toDelete);
    }
    toDelete.push_back(alloc);

    return true;
  }

  // Handle the cases which are the same for both array and struct (load/store)
  // then delegate any remaining to the derived class.
  void replaceUser(Operation *user, SmallVectorImpl<Operation *> &toDelete) {
    Derived *derived = getDerived();

    if (auto store = dyn_cast<StoreOp>(user)) {
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
        if constexpr (std::is_same_v<ContainerType, POP::StackAllocationOp>)
          break;
      }
      toDelete.push_back(user);
    } else if (isa<StackAllocLifetimeStartOp, StackAllocLifetimeEndOp>(user)) {
      llvm::BitVector eraseIndices(user->getNumOperands());
      for (auto [idx, value] : llvm::enumerate(user->getOperands())) {
        if (value == alloc)
          eraseIndices.set(idx);
      }
      user->eraseOperands(eraseIndices);
      user->insertOperands(user->getNumOperands(), newAllocas);
    } else {
      derived->replaceUserImpl(user, toDelete);
      toDelete.push_back(user);
    }
  }
};

class SROAStructLeafReplacer {
public:
  DebugInfo::DIExprParameterizedLeafReplacer<unsigned> direct;
  DebugInfo::DIExprParameterizedLeafReplacer<unsigned> indirect;

  SROAStructLeafReplacer()
      : direct(directLeafConversion), indirect(indirectLeafConversion) {}

private:
  /// Attempt to wrap leaves of the DI expression with AggregatesInto.
  /// Expects leaves to be of DIStructType.
  static ErrorOr<DebugInfo::DIExprAttr>
  directLeafConversion(DebugInfo::DIType irType, unsigned i) {
    auto structType = dyn_cast<DebugInfo::DIStructType>(irType);
    if (!structType)
      return Error("expected ir type to be a pointer to a struct type");

    // The element type of the struct is used directly.
    auto newElementType = structType.getMembers()[i].getType();

    // The leaf type is the struct element.
    auto newIrValue = DebugInfo::DIIRValueExprAttr::get(newElementType);
    // The expr is wrapped in an AggregatesInto expr to get back the
    // struct.
    auto aggregateExpr =
        DebugInfo::DIAggregatesIntoExprAttr::get(newIrValue, i, structType);

    return aggregateExpr;
  };

  /// Attempt to wrap leaves of the DI expression with AggregatesInto.
  /// Expects leaves to be a pointer to a DIStructType.
  static ErrorOr<DebugInfo::DIExprAttr>
  indirectLeafConversion(DebugInfo::DIType irType, unsigned i) {
    DebugInfo::DIType elementType;
    if (auto ptr = dyn_cast<DebugInfo::DIPointerType>(irType)) {
      elementType = ptr.getElementType();
    } else if (auto ptr = dyn_cast<DebugInfo::DITargetIndependentPointerType>(
                   irType)) {
      elementType = ptr.getElementType();
    } else {
      return Error("expected ir type to be a pointer type");
    }

    auto structType = dyn_cast<DebugInfo::DIStructType>(elementType);
    if (!structType)
      return Error("expected ir type to be a pointer to a struct type");

    // The element of the struct is immediately allocated into memory,
    // so we add a pointer type and wrap the expression with a deref.
    auto newElementType = DebugInfo::DITargetIndependentPointerType::get(
        structType.getMembers()[i].getType());

    // The leaf type is a pointer to the struct element.
    auto newIrValue = DebugInfo::DIIRValueExprAttr::get(newElementType);
    // The struct element was allocated to memory, so need to deref.
    auto derefExpr = DebugInfo::DIDerefExprAttr::get(newIrValue);
    // The expr is wrapped in an AggregatesInto expr to get back the
    // struct.
    auto aggregateExpr =
        DebugInfo::DIAggregatesIntoExprAttr::get(derefExpr, i, structType);
    // The address to the struct is obtained as the struct was
    // implicitly promoted.
    auto refExpr = DebugInfo::DIRefOfExprAttr::get(aggregateExpr, irType);

    return refExpr;
  };
};

/// The extra helper class for structures.
struct ReplaceStructs : public Replacer<ReplaceStructs, StructType> {
  using ContainerType = StructType;

  SROAStructLeafReplacer &leafReplacer;

  ReplaceStructs(OpBuilder &builder, StackAllocationOp alloc,
                 ContainerType container, uint32_t maxNumElements,
                 SROAStructLeafReplacer &leafReplacer,
                 SmallVector<Value> &valueMemCache)
      : Replacer(builder, alloc, container, maxNumElements, valueMemCache),
        leafReplacer(leafReplacer) {}

  bool canRun() {
    for (Operation *user : alloc->getUsers()) {
      // If the user is something which actually expects the full structure like
      // a call then we cannot perfom the optimization.
      if (!isa<StructGEPOp, POP::StoreOp, POP::LoadOp, DebugInfo::ValueOp,
               StackAllocLifetimeStartOp, StackAllocLifetimeEndOp>(user))
        return false;

      // If the user is the argument of the store, then we cannot elide.
      if (auto store = dyn_cast<POP::StoreOp>(user))
        if (store.getArg() == alloc)
          return false;
    }
    return true;
  }

  // Allocate the scalars which should replace the main alloc.
  void createScalarAllocs() {
    newAllocas.reserve(containerTy.getNumElements());
    for (Type elem : containerTy.getElementTypes()) {
      auto asPtr = PointerType::get(elem);
      Value v = builder.create<StackAllocationOp>(alloc.getLoc(), asPtr, 1);
      newAllocas.push_back(v);
    }
  }

  /// Replace some of the struct specific things.
  void replaceUserImpl(Operation *user,
                       SmallVectorImpl<Operation *> &toDelete) {
    if (auto gep = dyn_cast<StructGEPOp>(user)) {
      gep.replaceAllUsesWith(newAllocas[gep.getIndexAttr().getInt()]);
    } else if (auto load = dyn_cast<POP::LoadOp>(user)) {
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

      // Value of a loaded aggregate.
      Value loadedAgg;
      auto getOrCreateAggregateLoad = [&] {
        if (!loadedAgg) {
          for (unsigned i = 0, e = newAllocas.size(); i != e; ++i)
            (void)getOrCreateLoad(i);
          loadedAgg = builder.create<StructCreateOp>(
              load.getLoc(), load.getType(), loadedVals);
        }
        return loadedAgg;
      };

      // Replace the *user* of each load with the loaded scalar or for GEPs the
      // pointer itself.
      for (OpOperand &loadUser : llvm::make_early_inc_range(load->getUses())) {
        // Peephole `extract[i](load(%alloc))` -> `load(%newAlloc_i)`.
        if (auto extract = dyn_cast<StructExtractOp>(loadUser.getOwner())) {
          Value newVal = getOrCreateLoad(extract.getIndex());
          extract.replaceAllUsesWith(newVal);
          toDelete.push_back(extract);
        } else if (auto value =
                       dyn_cast<DebugInfo::ValueOp>(loadUser.getOwner())) {
          OpBuilder b(value);
          DebugInfo::DILocalVariableAttr valueInfo = value.getValueInfo();
          DebugInfo::DIExprAttr conversionExpr = value.getConversionExprAttr();
          for (auto [i, alloc] : llvm::enumerate(newAllocas)) {
            Value load = getOrCreateLoad(i);
            ErrorOr<DebugInfo::DIExprAttr> newConversionExpr =
                leafReplacer.direct.apply(conversionExpr, i);
            if (succeeded(newConversionExpr)) {
              b.create<DebugInfo::ValueOp>(value.getLoc(), load, valueInfo,
                                           newConversionExpr.get());
            }
          }
          toDelete.push_back(value);
        } else {
          // Replace any other value user with `create(load(%newAlloc_i), ...)`.
          loadUser.set(getOrCreateAggregateLoad());
        }
      }
    } else if (auto value = dyn_cast<DebugInfo::ValueOp>(user)) {
      OpBuilder b(value);
      DebugInfo::DILocalVariableAttr valueInfo = value.getValueInfo();
      DebugInfo::DIExprAttr conversionExpr = value.getConversionExprAttr();
      for (auto [i, alloc] : llvm::enumerate(newAllocas)) {
        ErrorOr<DebugInfo::DIExprAttr> newConversionExpr =
            leafReplacer.indirect.apply(conversionExpr, i);
        if (failed(newConversionExpr)) {
          // Not enough source information available to track this
          // transformation. Cannot debug this local variable anymore.
          continue;
        }
        b.create<DebugInfo::ValueOp>(value.getLoc(), alloc, valueInfo,
                                     newConversionExpr.get());
      }
    }
  }

  /// The extractor op for structures.
  Value createExtract(Location loc, Value operand, int64_t index) {
    return builder.create<StructExtractOp>(loc, operand,
                                           builder.getIndexAttr(index));
  }
};

/// The extra helper class for arrays.
struct ReplaceArray : public Replacer<ReplaceArray, POP::ArrayType> {
  using ContainerType = POP::ArrayType;

  ReplaceArray(OpBuilder &builder, StackAllocationOp alloc,
               ContainerType container, uint32_t maxNumElements,
               SmallVector<Value> &valueMemCache)
      : Replacer(builder, alloc, container, maxNumElements, valueMemCache) {}

  bool canRun() {
    // If we don't know the size of the array there's nothing to do.
    if (!containerTy.getResolvedSize())
      return false;

    // Don't decompose big arrays.
    if (containerTy.getResolvedSize() > maxNumElements)
      return false;

    for (Operation *user : alloc->getUsers()) {
      // If the user is something which actually expects the full structure like
      // a call then we cannot perfom the optimization.
      if (!isa<POP::ArrayGEPOp, POP::StoreOp, POP::LoadOp,
               StackAllocLifetimeStartOp, StackAllocLifetimeEndOp>(user))
        return false;

      // If the user is the argument of the store, then we cannot elide.
      if (auto store = dyn_cast<POP::StoreOp>(user))
        if (store.getArg() == alloc)
          return false;

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

    Type elem = containerTy.getElementType();
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
    } else if (auto load = dyn_cast<POP::LoadOp>(user)) {
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

      int64_t sizeOfArray = *containerTy.getResolvedSize();

      // Replace the *user* of each load with the loaded scalar or for GEPs the
      // pointer itself. We can't replace all users but we have several which
      // are easy cases to catch and help the compiler without requring
      // canonicalize & cse to be run again before mem2reg / more sroa.
      for (Operation *loadUser : llvm::make_early_inc_range(load->getUsers())) {
        if (auto get = dyn_cast<POP::ArrayGetOp>(loadUser)) {
          auto attr = cast<IntegerAttr>(get.getIndex());
          if (attr.getInt() < 0 || attr.getInt() > sizeOfArray)
            continue;

          Value newVal = getOrCreateLoad(attr.getInt());
          get.replaceAllUsesWith(newVal);
          get->dropAllReferences();
          toDelete.push_back(get);
        } else if (auto gep = dyn_cast<POP::ArrayGEPOp>(loadUser)) {
          APInt index;
          if (!matchPattern(gep.getIndex(), mlir::m_ConstantInt(&index)) ||
              index.isNegative())
            continue;

          // Oddly this comes up. Guard against out of range accesses.
          if (index.getSExtValue() >= sizeOfArray)
            continue;

          gep.replaceAllUsesWith(newAllocas[index.getLimitedValue()]);
          gep->dropAllReferences();
          toDelete.push_back(gep);
        }
      }

      // If there are any uses left materialize the array and use the
      // reconstituted array made up of each scalar aggregate inplace of the
      // load of the old one.
      if (!load.use_empty()) {
        // Load all the scalars.
        SmallVector<Value> allScalars;
        for (size_t i = 0; i < newAllocas.size(); ++i)
          allScalars.push_back(getOrCreateLoad(i));

        auto newArr = builder.create<POP::ArrayCreateOp>(
            load.getLoc(), load.getType(), allScalars);
        load.replaceAllUsesWith(newArr.getResult());
      }
    }
  }
};

/// In this case we treat the underlaying stack allocation as the container
/// itself.
struct ReplaceStack : public Replacer<ReplaceStack, POP::StackAllocationOp> {
  using ContainerType = POP::StackAllocationOp;

  ReplaceStack(OpBuilder &builder, StackAllocationOp alloc,
               uint32_t maxNumElements, SmallVector<Value> &valueMemCache)
      : Replacer(builder, alloc, alloc, maxNumElements, valueMemCache) {}

  bool canRun() {
    for (Operation *user : alloc->getUsers()) {
      if (!isa<POP::OffsetOp, POP::StoreOp, POP::LoadOp, POP::ArrayGEPOp,
               StructGEPOp, StackAllocLifetimeStartOp, StackAllocLifetimeEndOp>(
              user))
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

        // Offsets are aliases to the original stack allocation, not element
        // accesses. Require that all uses of the alias access the element.
        // We don't have to worry about handling constant offsets of constant
        // offsets, because they are canonicalized down to just one offset.
        for (Operation *user : offset->getUsers()) {
          if (!isa<POP::StoreOp, POP::LoadOp, POP::ArrayGEPOp, StructGEPOp>(
                  user))
            return false;
          // If the user is the argument of the store, then we cannot elide.
          if (auto store = dyn_cast<POP::StoreOp>(user))
            if (store.getArg() == offset)
              return false;
        }
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

    auto ptr = cast<PointerType>(alloc.getResult().getType());
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
    } else if (auto load = dyn_cast<LoadOp>(user)) {
      // A load from a stack allocation is implicitly the first element in the
      // stack.
      Value v = builder.create<POP::LoadOp>(load.getLoc(), newAllocas[0]);
      load.replaceAllUsesWith(v);
    }
  }
};

} // namespace

void SROAPass::runOnOperation() {
  OpBuilder builder{getOperation()->getContext()};

  SROAStructLeafReplacer leafReplacer;

  // The loop limit is an arbritary value to provide an upperbound on compile
  // time. However from experimentation this pass does not take a significant
  // amount of time to run and is a net-postive on compile time.
  constexpr size_t loopLimit = 10;

  SmallVector<Operation *, 32> toDelete;

  // The algorithm is serial so each step can use the same buffer backing which
  // avoids repeated allocations and deallocations.
  SmallVector<Value> valueMemCache;

  size_t iters = 0;

  bool changed = true;
  numReplacedAllocs = 0;
  while (changed && iters < loopLimit) {
    changed = false;
    iters++;
    toDelete.clear();

    getOperation()->walk([&](StackAllocationOp alloc) {
      // Skip non singleton stack allocations.
      auto count = dyn_cast<IntegerAttr>(alloc.getCount());
      if (!count)
        return;

      size_t numElems = count.getInt();

      // We won't try to decompose large stack allocations.
      if (numElems > maxNumElements)
        return;

      // Stack allocation is always a pointer to something.
      auto ptrType = cast<PointerType>(alloc.getResult().getType());

      // We decompose structs if there is one element otherwise we decompose the
      // stack itself.
      if (numElems != 1) {
        // Replace stack of N with N stacks of 1.
        ReplaceStack replacer{builder, alloc, maxNumElements, valueMemCache};
        changed |= replacer.run(toDelete);
      } else if (auto structTy =
                     dyn_cast<StructType>(ptrType.getElementType())) {
        ReplaceStructs replacer{builder,        alloc,        structTy,
                                maxNumElements, leafReplacer, valueMemCache};
        changed |= replacer.run(toDelete);
      } else if (auto arrayTy =
                     dyn_cast<POP::ArrayType>(ptrType.getElementType())) {
        ReplaceArray replacer{builder, alloc, arrayTy, maxNumElements,
                              valueMemCache};
        changed |= replacer.run(toDelete);
      }
    });

    // Delete the ops which are no longer used.
    numReplacedAllocs += toDelete.size();
    for (Operation *op : toDelete)
      op->erase();
  }

  // Control-flow is not modified.
  markAnalysesPreserved<HLCF::CFGAnalysis>();
}
