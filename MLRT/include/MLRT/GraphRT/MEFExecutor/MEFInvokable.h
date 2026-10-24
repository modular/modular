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

#ifndef MLRT_GRAPHRT_MEFEXECUTOR_MEFINVOKABLE_H
#define MLRT_GRAPHRT_MEFEXECUTOR_MEFINVOKABLE_H

#include "AsyncRT/ForwardDecls.h"
#include "Support/ErrorOr.h"
#include "Support/ForwardDecls.h"
#include "Support/FunctionExtras.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <functional>

namespace M {
class PrimitiveDescriptor;

/// This exposes an abstract interface that describes an "invokable", which is a
/// unit of computation with inputs and results.  It may be "invoked" which
/// triggers its bodies to start processing available dataflow values
/// asynchronously.
///
/// Invokables may be shared across many different runtimes.
///
class MEFInvokable {
public:
  virtual ~MEFInvokable() = default;

  size_t getNumArgs() const { return numArgs; }
  size_t getNumResults() const { return numResults; }

  /// Reflect over the body of the invokable, calling the visitor callback for
  /// each invokable. The visitor callback is provided with a StringRef that
  /// contains the primitive's name, and the PrimitiveDescriptor. This uses a
  /// C++-style function pointer so that the user can have some state passed
  /// into the function to, for example, collect all the primitives of a certain
  /// kind. This is not recursive, it only walks the primitives contained
  /// directly within this invokable. It can fail if the visitor callback fails,
  /// but it can also fail due to parsing errors so it's important to always
  /// check the return value.
  virtual ErrorOrSuccess walkPrimitives(
      std::function<ErrorOrSuccess(StringRef, const PrimitiveDescriptor &)>
          visitor) const = 0;

  /// The order in which to call the visitor while recursing over primitives.
  /// PreOrder indicates calling the visitor before visiting the invokables
  /// within the primitive, PostOrder indicates calling the visitor after
  /// visiting the invokables within the primitive.
  enum RecurseOrder {
    kPreOrder,
    kPostOrder,
  };
  /// Reflect over the body of the invokable recursively, calling the visitor
  /// callback for each invokable. The behavior of this is the same as
  /// walkPrimitives, but it will call the visitor on the current primitive
  /// either before or after (depending on the RecurseOrder) walking all the
  /// invokables found in each primitive.
  ErrorOrSuccess recursePrimitives(
      RecurseOrder order,
      std::function<ErrorOrSuccess(StringRef, const PrimitiveDescriptor &)>
          visitor) const;

  /// Reflect over the body of the invokable, calling the visitor callback for
  /// every object that has the same entity name as `T`. The entity must have 2
  /// static methods, one is a method called `getEntityName` that should return
  /// the primitive name that can be used for string matching. The second is
  /// called `getFromDescriptor`, which is a fallible static constructor. This
  /// static constructor should return `ErrorOr<T>`.
  template <typename T, typename Callable>
  ErrorOrSuccess walkPrimitives(Callable visitor) const {
    return walkPrimitives(
        [&](StringRef primitiveName,
            const PrimitiveDescriptor &descriptor) -> ErrorOrSuccess {
          // If the primitive name doesn't match the desired entity name, then
          // do nothing.
          if (primitiveName != T::getEntityName())
            return success();

          // Attempt to construct the object.
          auto objectOr = T::getFromDescriptor(descriptor);
          if (objectOr.isError())
            return objectOr.takeError();
          auto object = std::move(*objectOr);

          return invokeWithDefaultResultType<DefaultSuccess>(
              std::forward<Callable>(visitor), std::move(object));
        });
  }

  /// Reflect over the body of the invokable recursively, calling the visitor
  /// callback for every object that has the same entity name as `T`.
  template <typename T, typename Callable>
  ErrorOrSuccess recursePrimitives(RecurseOrder order, Callable visitor) const {
    return recursePrimitives(
        order,
        [&](StringRef primitiveName,
            const PrimitiveDescriptor &descriptor) -> ErrorOrSuccess {
          // If the primitive name doesn't match the desired entity name, then
          // do nothing.
          if (primitiveName != T::getEntityName())
            return success();

          // Attempt to construct the object.
          auto objectOr = T::getFromDescriptor(descriptor);
          if (objectOr.isError())
            return objectOr.takeError();
          auto object = std::move(*objectOr);

          return invokeWithDefaultResultType<DefaultSuccess>(
              std::forward<Callable>(visitor), std::move(object));
        });
  }

  /// Invoke this using the specified cpuDevice, passing the specified
  /// arguments. This fills in one AsyncValue for each result.  The caller
  /// is allowed to take from the `arguments` async values.
  virtual void invoke(MutableArrayRef<AnyAsyncValueRef> arguments,
                      MutableArrayRef<AnyAsyncValueRef> results,
                      CPUDevice &cpuDevice) const = 0;

  /// Invoke this using the specified cpuDevice, passing the specified
  /// arguments. This fills in one AsyncValue for each result.  The caller is
  /// allowed to take from the `arguments` async values.
  SmallVector<AnyAsyncValueRef, 1>
  invoke(MutableArrayRef<AnyAsyncValueRef> arguments,
         CPUDevice &cpuDevice) const;

  /// Add a new reference to this object.  The lifetime of invokables are
  /// generally tied to the file they are loaded from (e.g. a MEFFile) so they
  /// don't need explicit ownership management in most cases, but some cases
  /// that can extend their lifetime (e.g. implementations of async
  /// control flow operations) should extend their lifetime.
  virtual void addRef() const = 0;

  /// Drop a reference to this object, potentially deallocating it.
  virtual void dropRef() const = 0;

protected:
  MEFInvokable(uint16_t numArgs, uint16_t numResults)
      : numArgs(numArgs), numResults(numResults) {}

private:
  const uint16_t numArgs, numResults;

  MEFInvokable(MEFInvokable &) = delete;
  MEFInvokable &operator=(MEFInvokable &) = delete;
  virtual void vtableAnchor();
};

} // namespace M

#endif // MLRT_GRAPHRT_MEFEXECUTOR_MEFINVOKABLE_H
