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
// This file implements generic control flow primitives for the executor.
//
//===----------------------------------------------------------------------===//

#include "MLRT/GraphRT/Primitives.h"

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/GraphRT/MEFExecutor/MEFInvokable.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"

using namespace M;

//===----------------------------------------------------------------------===//
// grt.chain.create
//===----------------------------------------------------------------------===//

/// Strictly wait for all the input operands (if any), then return a ready
/// chain.
static AsyncValueRef<Chain> grt_chain_create(RemainingArguments arguments,
                                             CPUDevice &cpuDevice) {
  return cpuDevice.getReadyChain().copy();
}

//===----------------------------------------------------------------------===//
// grt.invoke
//===----------------------------------------------------------------------===//

static void grt_invoke(RemainingArguments arguments, RemainingResults results,
                       MEFAttr<MEFSymbolRef> targetName, MEFInvokable *target,
                       CPUDevice &cpuDevice) {
  // We don't use the attrs passed in - it's just the symbol name.
  assert(target->getNumArgs() == arguments.size() && "argument count mismatch");
  assert(target->getNumResults() == results.size() && "result count mismatch");
  target->invoke(arguments.get(), results.get(), cpuDevice);
}

//===----------------------------------------------------------------------===//
// grt.if
//===----------------------------------------------------------------------===//

/// This primitive dispatches to a 'true' or 'false' invokable based on a
/// condition.
///
/// Arguments: The first argument is the condition, with type i1/bool, and any
/// additional arguments are passed to the selected invokable.
///
/// trueInv/FalseInv:  The invokables must have matching signatures, and their
/// signatures must match grt.if's signature, exempting the extra i1 for the
/// condition.
static void grt_if(AsyncValueRef<bool> cond, RemainingArguments args,
                   RemainingResults results, MEFInvokable *trueInv,
                   MEFInvokable *falseInv, CPUDevice &cpuDevice) {
  assert(trueInv->getNumArgs() == args.size() && "argument count mismatch");
  assert(trueInv->getNumResults() == results.size() && "result count mismatch");
  assert(trueInv->getNumArgs() == falseInv->getNumArgs() &&
         trueInv->getNumResults() == falseInv->getNumResults() &&
         "true and false invokable types need to line up");

  // If the condition is already resolved and non-error, then avoid some
  // overhead.
  if (cond.getPointer()->isValueAvailable()) {
    MEFInvokable *bodyInv = *cond ? trueInv : falseInv;
    bodyInv->invoke(args.get(), results.get(), cpuDevice);
    return;
  }

  // Otherwise, go full async.  Define results as IndirectAsync values.
  results.allocateIndirectResults();

  struct IfState {
    RCRef<MEFInvokable> trueInvokable, falseInvokable;
    SmallVector<AnyAsyncValueRef> operands;
    SmallVector<AnyAsyncValueRef> ifResults;

    /// Emplaces all ifResults with a copy of errorResult.
    void fail(AnyAsyncValueRef &&errorResult) && {
      for (auto &result : ifResults)
        std::move(result).resolveIndirect(errorResult.copy());
    }

    /// This method is invoked when the AsyncValue for `cond` resolves.  The
    /// rest of the operands may still be unavailable, but results are filled
    /// in with the IndirectAsyncValue results to fill in.
    void runWhenCondResolves(AsyncValueRef<bool> &&cond) && {
      // If we have an error, then we can force propagate errors to all the
      // results.
      if (cond.isError())
        return std::move(*this).fail(std::move(cond));

      // Otherwise, execute the correct invokable based on the cond.
      // (Note we can't just pass ifResults for the invokable's results since
      // they are expected to be null).
      MEFInvokable *bodyInvokable =
          (*cond ? trueInvokable : falseInvokable).getPointer();
      SmallVector<AnyAsyncValueRef> bodyResults;
      bodyResults.resize(ifResults.size());
      bodyInvokable->invoke(operands, bodyResults, cond.getCPUDevice());

      // We complete by passing the final result values to the output of the
      // overall grt.if op.
      for (auto [ifResult, bodyResult] : llvm::zip(ifResults, bodyResults))
        std::move(ifResult).resolveIndirect(std::move(bodyResult));
    }
  };

  IfState state = {copyRCRef(trueInv), copyRCRef(falseInv), args.takeAsVector(),
                   results.copyAsVector()};

  std::move(cond).andThenSync(
      [state = std::move(state)](AsyncValueRef<bool> &&cond) mutable {
        std::move(state).runWhenCondResolves(std::move(cond));
      });
}

//===----------------------------------------------------------------------===//
// grt.while
//===----------------------------------------------------------------------===//

/// "Repeat while true" loop. Evaluate the 'cond' invokable, passing the current
/// value of the while arguments. If true, evaluate the 'body' invokable,
/// updating the current value of the while arguments. If false, return the
/// current value of the while arguments.
static void grt_while(RemainingArguments args, RemainingResults results,
                      MEFInvokable *cond, MEFInvokable *body,
                      CPUDevice &cpuDevice) {
  assert(args.size() == results.size());
  assert(cond->getNumArgs() == args.size() &&
         cond->getNumResults() == results.size() + 1);
  assert(body->getNumArgs() == args.size() &&
         body->getNumResults() == results.size());

  // Define results as IndirectAsync values.  The actual results are set in the
  // last iteration of the loop.
  results.allocateIndirectResults();

  /// Capture and add-ref each argument and result AsyncValue into a state we
  /// can manage.
  struct WhileState {
    CPUDevice &cpuDevice;
    RCRef<MEFInvokable> condInvokable;
    RCRef<MEFInvokable> bodyInvokable;
    SmallVector<AnyAsyncValueRef> currentOperands;
    SmallVector<AnyAsyncValueRef> whileResults;

    /// Emplaces all whileResults with a copy of errorResult.
    void fail(AnyAsyncValueRef &&errorResult) && {
      for (auto &result : whileResults)
        std::move(result).resolveIndirect(errorResult.copy());
    }

    /// Emplaces all whileResults with the currentOperands.
    void finish() && {
      for (auto [whileResult, currentOperand] :
           llvm::zip(whileResults, currentOperands))
        std::move(whileResult).resolveIndirect(std::move(currentOperand));
    }

    /// Evaluate the 'cond' block to decide if we need to execute the 'body'
    /// block (again).
    void evalCond() && {
      // Start executing the 'cond' block, consuming the current operands.
      SmallVector<AnyAsyncValueRef> condResults;
      condResults.resize(whileResults.size() + 1);
      condInvokable->invoke(currentOperands, condResults, cpuDevice);

      // When all the results of the 'cond' block complete, look at the
      // condition to decide if we should proceed to the 'body'. We could
      // just wait for the condition only, which would increase parallelism,
      // but that means we can have an unbounded number of sub-tasks and
      // greatly complicated async dataflow analysis.
      andThenSyncMoving(
          condResults,
          [state = std::move(*this)](
              MutableArrayRef<AnyAsyncValueRef> condResults) mutable {
            // The 'cond' block results (except the last condition result) are
            // the new current operands.
            for (auto [currentOperand, condResult] :
                 llvm::zip(state.currentOperands, llvm::drop_end(condResults)))
              currentOperand = std::move(condResult);

            AnyAsyncValueRef condResult = std::move(condResults.back());

            // If the condition yielded an error, give up.
            if (condResult.isError())
              return std::move(state).fail(std::move(condResult));
            // If the condition is false, we are done.
            if (!condResult.get<bool>())
              return std::move(state).finish();

            // Execute the body (and following cond) in a sub-task.
            addTask(state.cpuDevice, [state = std::move(state)]() mutable {
              std::move(state).evalBody();
            });
          });
    }

    /// Evaluate the 'body' block to update the currentOperands
    void evalBody() && {
      // Start executing the 'body' block, consuming the current operands.
      SmallVector<AnyAsyncValueRef> bodyResults;
      bodyResults.resize(whileResults.size());
      bodyInvokable->invoke(currentOperands, bodyResults, cpuDevice);

      // When all results of the 'body' block are ready, go around again
      // with the 'cond' block. We could directly proceed with the 'cond' block
      // without waiting for results, but that greatly complicates async
      // dataflow analysis.
      andThenSyncMoving(
          bodyResults,
          [state = std::move(*this)](
              MutableArrayRef<AnyAsyncValueRef> bodyResults) mutable {
            // The 'body' block results are the new current operands.
            for (auto [currentOperand, bodyResult] :
                 llvm::zip(state.currentOperands, bodyResults))
              currentOperand = std::move(bodyResult);

            // And we'll need to go around again to eval the condition.
            addTask(state.cpuDevice, [state = std::move(state)]() mutable {
              std::move(state).evalCond();
            });
          });
    }
  };

  WhileState state = {cpuDevice, copyRCRef(cond), copyRCRef(body),
                      args.takeAsVector(), results.copyAsVector()};

  // Start the ball rolling on a separate task.
  addTask(cpuDevice, [state = std::move(state)]() mutable {
    std::move(state).evalCond();
  });
}

//===----------------------------------------------------------------------===//
// grt.repeat.i32
//===----------------------------------------------------------------------===//

/// "Repeat N times" loop.  This takes a single int32_T iteration count, plus
/// arguments that are passed to the body and eventually returned.
///
/// This takes the count as an AsyncValue because this is a nonstrict
/// primitive, and the count may not be available yet.
static void repeat_i32(AsyncValueRef<int32_t> count, RemainingArguments args,
                       RemainingResults results, MEFInvokable *body,
                       CPUDevice &cpuDevice) {
  assert(args.size() == results.size() && args.size() == body->getNumArgs() &&
         results.size() == body->getNumResults());

  // Define results as IndirectAsync values.  The actual results are set in
  // the last iteration of the loop.
  results.allocateIndirectResults();

  /// Capture and add-ref each argument and result AsyncValue into a state we
  /// can manage.
  struct RepeatState {
    CPUDevice &cpuDevice;
    RCRef<MEFInvokable> bodyInvokable;
    SmallVector<AnyAsyncValueRef> currentOperands;
    SmallVector<AnyAsyncValueRef> repeatResults;

    /// This is the number of iterations left to run.
    int32_t iterationsLeft;

    /// Emplaces all repeatResults with a copy of errorResult.
    void fail(AnyAsyncValueRef &&errorResult) && {
      for (auto &result : repeatResults)
        std::move(result).resolveIndirect(errorResult.copy());
    }

    /// Emplaces all repeatResults with the currentOperands.
    void finish() && {
      for (auto [repeatResult, currentOperand] :
           llvm::zip(repeatResults, currentOperands))
        std::move(repeatResult).resolveIndirect(std::move(currentOperand));
    }

    /// This method is invoked when the AsyncValue for count resolves.  The
    /// rest of the operands may still be unavailable, but repeatResults are
    /// filled in with the IndirectAsyncValue results returned from the repeat
    /// loop overall.
    void runWhenCountResolves(AsyncValueRef<int32_t> &&count) && {
      // If we have an error, then we can force propagate errors to all the
      // results.
      if (count.isError())
        return std::move(*this).fail(std::move(count));

      // Now that count is resolved, we know how many iterations to do.
      iterationsLeft = *count;

      // Start executing the body.
      addTask(cpuDevice, [state = std::move(*this)]() mutable {
        std::move(state).doIteration();
      });
    }

    void doIteration() && {
      // If there are no iterations left, then the repeat results are
      // resolved to the outputs of the last iteration and we're done.
      if (iterationsLeft <= 0)
        // We complete by passing the final result values to the output of the
        // repeat loop overall.
        return std::move(*this).finish();

      // Otherwise, we have an iteration to perform.
      --iterationsLeft;

      SmallVector<AnyAsyncValueRef> bodyResults;
      bodyResults.resize(repeatResults.size());
      bodyInvokable->invoke(currentOperands, bodyResults, cpuDevice);

      // When all the results of the previous computation complete, kick off
      // the next iteration. We could just "unroll the loop" completely, which
      // would increase parallelism, but that means we can have an unbounded
      // number of sub-tasks and greatly complicated async dataflow analysis.
      andThenSyncMoving(
          bodyResults, [state = std::move(*this)](
                           MutableArrayRef<AnyAsyncValueRef> values) mutable {
            // bodyResults become the operands of the next iteration.
            for (auto [currentOperand, value] :
                 llvm::zip(state.currentOperands, values))
              currentOperand = std::move(value);

            addTask(state.cpuDevice, [state = std::move(state)]() mutable {
              std::move(state).doIteration();
            });
          });
    }
  };

  RepeatState state = {cpuDevice, copyRCRef(body), args.takeAsVector(),
                       results.copyAsVector(), 0};

  // Dispatch when the `count` becomes available (possibly immediately).
  std::move(count).andThenSync(
      [state = std::move(state)](AnyAsyncValueRef &&count) mutable {
        std::move(state).runWhenCountResolves(std::move(count));
      });
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// This registers the 'graphrt' dialect primitives for control flow.
void M::registerGraphRTControlFlowPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "grt.chain.create", grt_chain_create);
  ADD_PRIMITIVE(registry, "grt.invoke", grt_invoke);
  ADD_PRIMITIVE(registry, "grt.if", grt_if);
  ADD_PRIMITIVE(registry, "grt.while", grt_while);
  ADD_PRIMITIVE(registry, "grt.repeat.i32", repeat_i32);
}
