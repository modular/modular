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
// This is the implementation logic for the MEFExecutor.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_MEFEXECUTOR_MEFFILEIMPL_H
#define MLRT_GRAPHRT_MEFEXECUTOR_MEFFILEIMPL_H

#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "MLRT/GraphRT/MEFExecutor/MEFInvokable.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "llvm/ADT/ArrayRef.h"

namespace M {
class MEFInvokableImpl;

/// This class is the implementation details behind the MEFFile::Open method,
/// which maintains all the state necessary for the MEFExecutor.
class MEFFileImpl final : public MEFFile {
public:
  MEFFileImpl() = default;
  ~MEFFileImpl() override;

  /// Return the invokable with the specified invokable ID.
  MEFInvokable *getInvokableByID(size_t number) const;

  /// This method enumerates the primitive name table, calling the specified
  /// callable on each entry.  The callable function may return an error to stop
  /// iteration and diagnose problems, if it does, it is propagated out.
  ErrorOrSuccess enumeratePrimitiveNames(
      const std::function<ErrorOrSuccess(StringRef)> &callbackFn);

  // Everything in this class is public because the entire class is private to
  // the MEFExecutor library implementation.
public:
  // This struct describes one invokable in the InvokableIndex section.
  struct InvokableIndex {
    uint32_t offsetInInvokableBodiesSection;
    uint32_t offsetOfNameInStringSection;
    uint32_t offsetOfLocation;
    uint16_t numArgs, numResults;
  };

  // These are sections that we expect to be in the MEF file.  They are lazily
  // decoded when executing invokables.
  ArrayRef<uint8_t> primitiveNames;
  ArrayRef<uint8_t> locationFilenames;
  ArrayRef<uint8_t> locationPositions;
  ArrayRef<uint8_t> strings;
  ArrayRef<uint8_t> attributes;
  ArrayRef<uint8_t> invokableBodies;

  /// This is the list of primitive implementations and their decoded names
  /// arranged by their ID.
  std::vector<PrimitiveRegistry::PrimitiveImplFn> primitives;
  std::vector<StringRef> decodedPrimitiveNames;

  /// This contains all of the MEFInvokable records inline in dense memory even
  /// though clients treat them like individual objects.  Their lifetime is
  /// completely bound to MEFFileImpl's.
  MEFInvokableImpl *invokables = nullptr;
  size_t numInvokables = 0;
};

/// MEF specific implementation of the Invokable interface.
class MEFInvokableImpl final : public MEFInvokable, public LocationDecoder {
public:
  MEFInvokableImpl(uint16_t numArguments, uint16_t numResults,
                   uint32_t bodyOffset, MEFFileImpl *owner)
      : MEFInvokable(numArguments, numResults), bodyOffset(bodyOffset),
        owner(owner) {}

  ~MEFInvokableImpl() override = default;

  /// Visit each primitive in this invokable.
  ErrorOrSuccess walkPrimitives(
      std::function<ErrorOrSuccess(StringRef, const PrimitiveDescriptor &)>
          visitor) const override;

  /// Invokable this using the specified cpuDevice, passing the specified
  /// arguments. This returns one AsyncValue for each result.
  void invoke(MutableArrayRef<AnyAsyncValueRef> arguments,
              MutableArrayRef<AnyAsyncValueRef> results,
              CPUDevice &cpuDevice) const override;

  /// Add a new reference to this object.  The lifetime of invokables are
  /// generally tied to the file they are loaded from (e.g. a MEFFile) so they
  /// don't need explicit ownership management in most cases, but some cases
  /// that can extend their lifetime (e.g. implementations of async
  /// control flow operations) should extend their lifetime.
  void addRef() const override;

  /// Drop a reference to this object, potentially deallocating it.
  void dropRef() const override;

  /// Implement the only remaining LocationDecoder hook. Since we have a pointer
  /// to the MEFFileImpl we can easily decode a location provided to us.
  M::AsyncRT::DecodedLocation decode(const EncodedLocation &loc) const override;

public:
  const uint32_t bodyOffset;
  MEFFileImpl *const owner;
};

} // namespace M

#endif // MLRT_GRAPHRT_MEFEXECUTOR_MEFFILEIMPL_H
