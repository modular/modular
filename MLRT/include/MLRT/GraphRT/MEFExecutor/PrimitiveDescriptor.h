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

#ifndef MLRT_GRAPHRT_MEFEXECUTOR_PRIMITIVEDESCRIPTOR_H
#define MLRT_GRAPHRT_MEFEXECUTOR_PRIMITIVEDESCRIPTOR_H

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/AsyncValue.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace M {
class MEFInvokable;
template <typename T>
class MEFAttr;

/// This class captures the states associated with an invocation of a primitive,
/// including the input arguments, attributes, result values, location and
/// cpuDevice.
///
/// The implementation of a primitive is responsible for creating AsyncValue
/// objects and setting them into the frame.  The correct number of results are
/// present, but each AsyncValue pointers is null when the implementation is
/// invoked.
class PrimitiveDescriptor {
public:
  /// Return an EncodedLocation for this specified frame invocation.
  EncodedLocation copyLocation() const {
    return EncodedLocation(locationData, copyRCRef(&locationDecoder));
  }

  //===--------------------------------------------------------------------===//
  // Arguments
  //===--------------------------------------------------------------------===//

  size_t getNumArgs() const { return numArguments; }

  //===--------------------------------------------------------------------===//
  // Attributes
  //===--------------------------------------------------------------------===//

  /// Get the number of attributes.
  size_t getNumAttrs() const { return attrOffsets.size(); }

  /// Get the attribute at the given index.
  const uint8_t *getRawAttr(size_t index) const {
    return attrSection.data() + attrOffsets[index];
  }

  /// Define an SFINAE helper to unwrap a type T that may or may not be wrapped
  /// in a MEFAttr<T>.
  template <typename T>
  struct UnwrapMEFAttr {
    using InnerType = T;
  };
  template <typename T>
  struct UnwrapMEFAttr<MEFAttr<T>> {
    using InnerType = T;
  };

  /// Get a MEFAttr<T> at the given index.
  template <typename T>
  auto getAttr(size_t index) const {
    return MEFAttr<typename UnwrapMEFAttr<T>::InnerType>(getRawAttr(index),
                                                         attrSection);
  }

  /// Return all the offsets for all attributes in this primitive frame.
  ArrayRef<uint64_t> getAttrOffsets() const { return attrOffsets; }
  const uint8_t *getAttrBase() const { return attrSection.data(); }
  ArrayRef<uint8_t> getAttrSection() const { return attrSection; }

  //===--------------------------------------------------------------------===//
  // Invokables
  //===--------------------------------------------------------------------===//

  size_t getNumInvokables() const { return invokables.size(); }

  MEFInvokable *getInvokable(size_t idx) const {
    assert(idx < invokables.size() && "invalid invokable number");
    return invokables[idx];
  }

  //===--------------------------------------------------------------------===//
  // Results
  //===--------------------------------------------------------------------===//

  size_t getNumResults() const { return numResults; }

protected:
  PrimitiveDescriptor(ArrayRef<uint8_t> attrSection,
                      LocationDecoder &locationDecoder)
      : locationDecoder(locationDecoder), attrSection(attrSection) {}

  LocationDecoder &locationDecoder;

  /// This is the overall attribute section in the MEF file.
  const ArrayRef<uint8_t> attrSection;

  /// This is the location information for the current frame, which (when
  /// combined with the location decoder) can be turned into an
  /// `EncodedLocation`.
  intptr_t locationData = ~intptr_t(0);

  /// numResults is set to ~0 so we can check that addArgument() is called
  /// before setNumResults.
  uint32_t numArguments = 0, numResults = uint32_t(~0U);

  /// This is all of the invokable references used by the invocation.
  SmallVector<MEFInvokable *, 2> invokables;

  /// This keeps track of each attribute value into the attributeSection.
  ArrayRef<uint64_t> attrOffsets;
};

/// This class is used to construct a PrimitiveDescriptor object without
/// exposing the builder methods to the implementation functions.
///
class PrimitiveDescriptorBuilder : public PrimitiveDescriptor {
public:
  PrimitiveDescriptorBuilder(ArrayRef<uint8_t> attrSection,
                             LocationDecoder &locationDecoder)
      : PrimitiveDescriptor(attrSection, locationDecoder) {}

  void reset() {
    locationData = ~intptr_t(0);
    numArguments = 0;
    numResults = uint32_t(~0U);
    attrOffsets = {};
    invokables.clear();
  }

  void setLocationData(intptr_t locationData) {
    this->locationData = locationData;
  }

  /// Set the offsets to the attributes in the MEF attribute section.
  void setAttrs(ArrayRef<uint64_t> attrOffsets) {
    this->attrOffsets = attrOffsets;
  }

  void addInvokable(MEFInvokable *inv) { invokables.push_back(inv); }
};

/// This is the primitive descriptor you can actually run - it holds the
/// AsyncValue objects that are needed to run a primitive.
class RunnablePrimitiveDescriptor : public PrimitiveDescriptor {
public:
  CPUDevice &getRuntime() const { return cpuDevice; }

  //===--------------------------------------------------------------------===//
  // Arguments
  //===--------------------------------------------------------------------===//

  /// Get the argument at the given index as AsyncValue*.
  AnyAsyncValueRef &getArg(size_t index) {
    assert(index < getNumArgs() && "invalid argument number");
    return argsAndResults[index];
  }

  /// Look through the argument AsyncValue to get the element as type T.
  template <typename T>
  T &getArg(size_t index) {
    return getArg(index).get<T>();
  }

  ArrayRef<AnyAsyncValueRef> getArguments() const {
    return ArrayRef(argsAndResults.data(), getNumArgs());
  }

  MutableArrayRef<AnyAsyncValueRef> getArguments() {
    return llvm::MutableArrayRef(argsAndResults.data(), getNumArgs());
  }

  //===--------------------------------------------------------------------===//
  // Results
  //===--------------------------------------------------------------------===//

  ArrayRef<AnyAsyncValueRef> getResults() const {
    return const_cast<RunnablePrimitiveDescriptor *>(this)->getResults();
  }

  MutableArrayRef<AnyAsyncValueRef> getResults() {
    AnyAsyncValueRef *firstResult = argsAndResults.data() + numArguments;
    return llvm::MutableArrayRef(firstResult, getNumResults());
  }

  /// Set the result at the given index with the given AsyncValue.
  void setResult(size_t index, AnyAsyncValueRef value) {
    AnyAsyncValueRef &result = getResults()[index];
    assert(!result && "result is already set");
    result = std::move(value);
  }

  /// Set the result at the given index to a newly constructed T value with the
  /// specified constructor arguments.
  template <typename T, typename... Args>
  void emplaceResult(size_t index, Args &&...args) {
    setResult(index, AsyncValueRef<T>::createReady(
                         cpuDevice, std::forward<Args>(args)...));
  }

  //===--------------------------------------------------------------------===//
  // Error Results
  //===--------------------------------------------------------------------===//

  /// Create an AsyncValue with the specified cause, and our current location.
  /// This can be set into a result or used for other purposes.
  AnyAsyncValueRef getError(Error cause);

  /// Mark that execution of the primitive failed with the specified reason.
  /// This fills in all remaining unspecified results with that error value.
  void fail(Error cause);

protected:
  RunnablePrimitiveDescriptor(CPUDevice &cpuDevice,
                              ArrayRef<uint8_t> attrSection,
                              LocationDecoder &locationDecoder)
      : PrimitiveDescriptor(attrSection, locationDecoder),
        cpuDevice(cpuDevice) {}

  CPUDevice &cpuDevice;

  /// We store arguments and results in a single SmallVector for efficiency:
  /// one less vector to overflow.
  SmallVector<AnyAsyncValueRef, 16> argsAndResults;
};

/// This class is used to construct a RunnablePrimitiveDescriptor object without
/// exposing the builder methods to the implementation functions.
///
/// When initializing a RunnablePrimitiveDescriptor, this class requires that
/// the client performs the following actions in order:
///   1. Add argument values.
///   2. Set the number of results with `setNumResults`.
///   3. Add attributes values.
///
class RunnablePrimitiveDescriptorBuilder : public RunnablePrimitiveDescriptor {
public:
  RunnablePrimitiveDescriptorBuilder(CPUDevice &cpuDevice,
                                     ArrayRef<uint8_t> attrSection,
                                     LocationDecoder &locationDecoder)
      : RunnablePrimitiveDescriptor(cpuDevice, attrSection, locationDecoder) {}

  /// Clear all fields.
  void reset() {
    locationData = ~intptr_t(0);
    numArguments = 0;
    numResults = uint32_t(~0U);
    attrOffsets = {};
    invokables.clear();
    argsAndResults.clear();
  }

  void setLocationData(intptr_t locationData) {
    this->locationData = locationData;
  }

  /// Reserve space in the arg/result vector to reduce allocations.
  void reserveArgsAndResults(size_t numArgs, size_t numResults) {
    argsAndResults.reserve(numArgs + numResults);
  }

  /// Get mutable access to the argument list.
  MutableArrayRef<AnyAsyncValueRef> getArguments() {
    return llvm::MutableArrayRef(argsAndResults.data(), getNumArgs());
  }

  /// Add a new argument.
  void addArg(AnyAsyncValueRef value) {
    assert(numResults == uint32_t(~0U) &&
           "Must call AddArg before calling SetNumResults");
    argsAndResults.push_back(std::move(value));
    ++numArguments;
  }

  /// Set the number of results expected, and allocate space to capture them.
  void setNumResults(size_t n) {
    assert(numArguments == argsAndResults.size() &&
           numResults == uint32_t(~0U));
    numResults = n;
    argsAndResults.resize(argsAndResults.size() + n);
  }

  AnyAsyncValueRef &getResult(size_t index) {
    assert(index < numResults && "invalid result #");
    return argsAndResults[numArguments + index];
  }

  /// Set the offsets to the attributes in the MEF attribute section.
  void setAttrs(ArrayRef<uint64_t> attrOffsets) {
    this->attrOffsets = attrOffsets;
  }

  void addInvokable(MEFInvokable *inv) { invokables.push_back(inv); }
};

} // namespace M

#endif // MLRT_GRAPHRT_MEFEXECUTOR_PRIMITIVEDESCRIPTOR_H
