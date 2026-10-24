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

#ifndef MLRT_GRAPHRT_MEFEXECUTOR_PRIMITIVEREGISTRY_H
#define MLRT_GRAPHRT_MEFEXECUTOR_PRIMITIVEREGISTRY_H

#include "AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/GraphRT/MEF/MEFAttr.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveDescriptor.h"
#include "Support/ErrorOr.h"
#include "Support/Profiling/TimeProfiler.h"

namespace M {

//===----------------------------------------------------------------------===//
// PrimitiveProfilerEntry
//===----------------------------------------------------------------------===//

/// Profiling entry for *all* GraphRT primitives, irrespective of dialect.
using PrimitiveProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kPrimitives, 1),
                  Trace::kPrimitives>;

//===----------------------------------------------------------------------===//
// PrimitiveRegistry
//===----------------------------------------------------------------------===//

/// This represents a mapping between the names of the MLIR opcodes to the
/// implementations of those functions, along with type mappings.
class PrimitiveRegistry final {
public:
  PrimitiveRegistry();
  ~PrimitiveRegistry();

  /// Primitive implementations use this signature, acting on the data provided
  /// in the argument frame, and returning results through it.
  using PrimitiveImplFn = void (*)(RunnablePrimitiveDescriptor &frame);

  /// Add an implementation of a primitive to the registry.  Consider using
  /// the ADD_PRIMITIVE macro to enable various pretty forms for this.
  void add(StringLiteral name, PrimitiveImplFn fn);

  /// Finalize the registry. Calling `get()` before the registry is finalized
  /// will result in undefined behavior.
  void finalize();

  /// Look up a primitive with a specified name out of the registry, returning
  /// null if no match is found.
  PrimitiveImplFn get(StringRef name) const;

private:
  PrimitiveRegistry(const PrimitiveRegistry &) = delete;
  PrimitiveRegistry &operator=(const PrimitiveRegistry &) = delete;

  struct Impl;
  std::unique_ptr<Impl> impl;
};

//===----------------------------------------------------------------------===//
// Helper Classes For Primitive Implementations
//===----------------------------------------------------------------------===//

// Utility template for tag dispatching.
template <typename... T>
struct TypeTag {};

/// RemainingArguments collects all remaining arguments in an ArrayRef. There
/// can be at most one RemainingArguments/RepeatedArguments, and it must appear
/// after all other Arguments.
//
/// This should only be used when argument types have potentially heterogenous
/// types, for example GRT::InvokeOp.
class RemainingArguments {
public:
  explicit RemainingArguments(MutableArrayRef<AnyAsyncValueRef> arguments)
      : arguments(arguments) {}

  MutableArrayRef<AnyAsyncValueRef> get() const { return arguments; }

  bool empty() const { return arguments.empty(); }
  size_t size() const { return arguments.size(); }
  AnyAsyncValueRef &operator[](size_t i) const { return arguments[i]; }

  using iterator = MutableArrayRef<AnyAsyncValueRef>::iterator;
  iterator begin() const { return arguments.begin(); }
  iterator end() const { return arguments.end(); }

  /// Return the AsyncValues as a SmallVector where every element is moved out.
  SmallVector<AnyAsyncValueRef> takeAsVector() {
    SmallVector<AnyAsyncValueRef> result;
    result.reserve(arguments.size());
    for (auto &a : arguments)
      result.push_back(std::move(a));
    return result;
  }

  /// Return the AsyncValues as a std::vector where every element is moved out.
  std::vector<AnyAsyncValueRef> takeAsStdVector() {
    std::vector<AnyAsyncValueRef> result;
    result.reserve(arguments.size());
    for (auto &a : arguments)
      result.push_back(std::move(a));
    return result;
  }

  /// Return the AsyncValues as a SmallVector where every element is copied out.
  SmallVector<AnyAsyncValueRef> copyAsVector() {
    SmallVector<AnyAsyncValueRef> result;
    result.reserve(arguments.size());
    for (auto &a : arguments)
      result.push_back(a.copy());
    return result;
  }

private:
  MutableArrayRef<AnyAsyncValueRef> arguments;
};

/// This is a common base class between Result and IndirectResult, to reduce
/// code duplication.
class AnySingleResult {
public:
  /// On construction, the result is initialized with an AsyncValue of the
  /// correct type.
  explicit AnySingleResult(AnyAsyncValueRef &result,
                           RunnablePrimitiveDescriptor &frame)
      : result(result), frame(frame) {
    assert(!result && "result already initialized?");
  }
  AnySingleResult(const AnySingleResult &&other)
      : result(other.result), frame(other.frame) {
    // NOTE(#7399): Should probably change result and frame to be pointers and
    // null them out to detect use-after-move mistakes.
  }

  /// Return a fresh reference to the AsyncValue allocated to hold the result.
  AnyAsyncValueRef copy() const { return result.copy(); }

  /// Return a copy of the location for the currently execution primitive.
  EncodedLocation copyLocation() const { return getFrame().copyLocation(); }

  /// Fill this AsyncValue with an error that has the specified message
  /// and the current frame's location.
  void setToError(Error message) && {
    // CAUTION: We must copy result since it is expected to remain available
    // to the creator of the AnySingleResult. However, we'll still mark this
    // method as moving to make clear the result should only be set once.
    result.copy().setToError({std::move(message), frame.copyLocation()});
  }

  /// Fill this AsyncValue with an error that has the specified diagnostic.
  void setToError(EncodedDiagnostic diagnostic) && {
    // CAUTION: We must copy result since it is expected to remain available
    // to the creator of the AnySingleResult. However, we'll still mark this
    // method as moving to make clear the result should only be set once.
    result.copy().setToError(std::move(diagnostic));
  }

  RunnablePrimitiveDescriptor &getFrame() const { return frame; }

  bool isError() const { return result.isError(); }

protected:
  AnyAsyncValueRef &result;

private:
  AnySingleResult(const AnySingleResult &) = delete;
  void operator=(const AnySingleResult &) = delete;
  RunnablePrimitiveDescriptor &frame;
};

/// The Result class is used by primitive implementations for typed asynchronous
/// results.
template <typename T>
class Result : public AnySingleResult {
public:
  explicit Result(AnyAsyncValueRef &result, RunnablePrimitiveDescriptor &frame)
      : AnySingleResult(result, frame) {
    result = AnyAsyncValueRef::allocate<T>(frame.getRuntime());
  }
  using AnySingleResult::AnySingleResult;

  /// Return a fresh reference to the AsyncValue allocated to hold the result.
  /// when copying into a closure.
  AsyncValueRef<T> copy() const {
    return AsyncValueRef<T>(AnySingleResult::copy());
  }

  bool isError() const { return AnySingleResult::isError(); }

  /// Return an owning reference to the AsyncValueRef for this result, along
  /// with an EncodedLocation that allows for elegant async error reporting.
  AsyncRT::AsyncValueRefWithEncodedLocation<AsyncValueRef<T>>
  copyWithLocation() const {
    return {copy(), copyLocation()};
  }

  /// Return a reference to the underlying element in the AsyncValue if it
  /// is constructed.
  T &get() const { return result.template get<T>(); }

  /// Construct the result in place and mark it as ready.
  template <typename... Args>
  void emplace(Args &&...args) && {
    // CAUTION: We must copy result since it is expected to remain available
    // to the creator of the AnySingleResult. However, we'll still mark this
    // method as moving to make clear the result should only be set once.
    result.copy().template emplace<T, Args...>(std::forward<Args>(args)...);
  }
};

/// The IndirectResult class is used for primitives that need a result but don't
/// know its static type.  It will be initialized to an AsyncValue that is in
/// indirect state, allowing it to be late-bound, e.g. asynchronously.
class IndirectResult : public AnySingleResult {
public:
  /// On construction, this is initialized with an indirect AsyncValue.
  explicit IndirectResult(AnyAsyncValueRef &result,
                          RunnablePrimitiveDescriptor &frame)
      : AnySingleResult(result, frame) {
    result = AnyAsyncValueRef::createIndirect(frame.getRuntime());
  }
  using AnySingleResult::AnySingleResult;

  /// Return an owning reference to the AsyncValueRef for this result, along
  /// with an EncodedLocation that allows for elegant async error reporting.
  AsyncRT::AsyncValueRefWithEncodedLocation<AnyAsyncValueRef>
  copyWithLocation() const {
    return {copy(), copyLocation()};
  }

  /// Resolve an IndirectAsyncValue to point to the specified new value,
  /// resolving any waiters whenever newValue becomes ready.
  void resolve(AnyAsyncValueRef newValue) && {
    // CAUTION: We must copy result since it is expected to remain available
    // to the creator of the AnySingleResult. However, we'll still mark this
    // method as moving to make clear the result should only be set once.
    result.copy().resolveIndirect(newValue.releaseRCRef());
  }

  /// Construct an AsyncValue for the result and resolve the indirect result.
  template <typename T, typename... Args>
  void emplace(Args &&...args) && {
    std::move(*this).resolve(AnyAsyncValueRef::createReady<T, Args...>(
        result.getCPUDevice(), std::forward<Args>(args)...));
  }
};

/// RemainingResults collects all remaining results in a MutableArrayRef. There
/// can be at most one RemainingResults, and it must appear after all other
/// Results.
///
/// This should be used when a primitive has potentially heterogenous types, for
/// example, GRT::InvokeOp.
class RemainingResults {
public:
  explicit RemainingResults(MutableArrayRef<AnyAsyncValueRef> results,
                            CPUDevice &cpuDevice)
      : cpuDevice(cpuDevice), results(results) {}

  /// Returns reference to underlying results array.
  /// CAUTION: The results are shared between the producer (eg a primitive)
  /// and consumer (the creator of RemainingResults), so generally should
  /// not be moved.
  MutableArrayRef<AnyAsyncValueRef> get() const { return results; }
  bool empty() const { return results.empty(); }
  size_t size() const { return results.size(); }
  AnyAsyncValueRef &operator[](size_t i) const { return results[i]; }

  using iterator = MutableArrayRef<AnyAsyncValueRef>::iterator;
  iterator begin() const { return results.begin(); }
  iterator end() const { return results.end(); }

  /// Allocate all of the unset result values as IndirectAsyncValues.
  void allocateIndirectResults() {
    for (auto &elt : results)
      if (!elt)
        elt = AnyAsyncValueRef::createIndirect(cpuDevice);
  }

  /// Return the AsyncValues as a SmallVector where every element is copied out.
  SmallVector<AnyAsyncValueRef> copyAsVector() {
    SmallVector<AnyAsyncValueRef> result;
    result.reserve(results.size());
    for (auto &r : results)
      result.push_back(r.copy());
    return result;
  }

  /// Set all results in this container to error with message `error` and
  /// location `loc`.
  void setAllToError(Error error, const EncodedLocation &loc) && {
    for (auto &result : results)
      result = AnyAsyncValueRef::createError(
          cpuDevice, EncodedDiagnostic(error.copy(), loc.copy()));
  }

private:
  CPUDevice &cpuDevice;
  MutableArrayRef<AnyAsyncValueRef> results;
};

// Allow decoding MEFListAttr into a SmallVector.
template <typename EltType, size_t Size>
struct MEFDecoder<SmallVector<EltType, Size>> {
  static ErrorOr<SmallVector<EltType, Size>> decode(MEFListAttr value) {
    SmallVector<EltType, Size> result;
    result.reserve(value.size());
    for (auto element : value) {
      EltType resultElt;
      if (auto err = element.decode(&resultElt))
        return err.takeError();
      result.push_back(resultElt);
    }
    return result;
  }
};

/// RemainingMEFAttrs collects all remaining attributes for the primitive that
/// were not explicitly handled. There can be at most one RemainingMEFAttrs,
/// and it must appear after all other attributes.
class RemainingMEFAttrs {
public:
  explicit RemainingMEFAttrs(const uint8_t *basePtr, ArrayRef<uint64_t> attrs,
                             ArrayRef<uint8_t> attrSection)
      : basePtr(basePtr), attrs(attrs), attrSection(attrSection) {}

  bool empty() const { return attrs.empty(); }
  size_t size() const { return attrs.size(); }
  MEFDynamicAttr operator[](size_t i) const {
    return MEFDynamicAttr(basePtr + attrs[i], attrSection);
  }

private:
  const uint8_t *basePtr;
  ArrayRef<uint64_t> attrs;
  ArrayRef<uint8_t> attrSection;
};

//===----------------------------------------------------------------------===//
// Primitive Registration Wrapper Generator
//===----------------------------------------------------------------------===//

/// This macro is used to register an implementation function for a primitive
/// with a specified PrimitiveRegistry.  It handles generating a wrapper thunk
/// that makes it much easier (and less bug prone) to work with low-level
/// `RunnablePrimitiveDescriptor`.
///
/// For example:
///    ADD_PRIMITIVE(registry, "grt.invoke", grt_invoke_impl);
///
#define ADD_PRIMITIVE(registry, name, fn)                                      \
  do {                                                                         \
    (registry).add(                                                            \
        name, ::M::Detail::PrimitiveImpl<decltype(&(fn)), &(fn)>::invoke);     \
  } while (0)

namespace Detail {
// This class implements the ADD_PRIMITIVE macro.
template <typename FnType, FnType implFn>
struct PrimitiveImpl;

template <typename Return, typename... Args, Return (*implFn)(Args...)>
struct PrimitiveImpl<Return (*)(Args...), implFn> {

  //===--------------------------------------------------------------------===//
  // Invocation
  //===--------------------------------------------------------------------===//

  /// This is the main entry point that gets registered as an impl.  This works
  /// by recursively processing the argument list, then handling the result
  /// list, generating a function with the standard PrimitiveImpl signature that
  /// handles each part of the function signature as it goes.
  static void invoke(RunnablePrimitiveDescriptor &frame) {
    processArgsAndInvoke<
        ValueIndices</*argIdx=*/0, /*resultIdx=*/0, /*attrIdx=*/0,
                     /*invIdx=*/0>,
        /*hasInChain=*/false>(TypeTag<Args...>(), frame);
  }

  //===--------------------------------------------------------------------===//
  // Argument handling
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Argument handling: Value Type Definitions

  /// This enum represents the different types of values that may be passed as
  /// arguments during the invocation of a primitive.
  enum class ValueType {
    kArgument,
    kResult,
    kAttribute,
    kInvokable,
    kFramework,
  };

  /// This class represents a static container for the various indices used
  /// during argument handling.
  template <int argIdxVal, int resultIdxVal, int attrIdxVal, int invIdxVal>
  struct ValueIndices {
    /// A set of indices that correspond to lists of various value types. When
    /// each list has been fully consumed, the index is set to -1.
    static constexpr int argIdx = argIdxVal;
    static constexpr int resultIdx = resultIdxVal;
    static constexpr int attrIdx = attrIdxVal;
    static constexpr int invIdx = invIdxVal;
  };

  /// This class provides a common static abstraction of the different value
  /// types.
  template <ValueType valueTypeParam, bool valueIsTerminal = false>
  struct ValueBase {
    /// A flag indicating the type of the value.
    static constexpr ValueType valueType = valueTypeParam;

    /// A flag indicating whether the value is a terminal value, i.e. it
    /// consumes the remaining values of this type.
    static constexpr bool isTerminal = valueIsTerminal;

    /// A flag indicating whether the value contains a chain.
    static constexpr bool isChain = false;

    /// Updates the given index if the type matches the one for this value,
    /// otherwise returns the current index.
    template <ValueType type, int idx>
    static constexpr int updateIndex() {
      return (valueType == type) ? (isTerminal ? -1 : (idx + 1)) : idx;
    }
    /// Updates the given indices for the current value.
    template <typename IndicesT>
    using UpdatedIndicesT =
        ValueIndices<updateIndex<ValueType::kArgument, IndicesT::argIdx>(),
                     updateIndex<ValueType::kResult, IndicesT::resultIdx>(),
                     updateIndex<ValueType::kAttribute, IndicesT::attrIdx>(),
                     updateIndex<ValueType::kInvokable, IndicesT::invIdx>()>;

    /// Verify the given value indices based on the current value type. The
    /// default implementation assumes all are valid, but derived value types
    /// may override to provide additional checks and guarantees.
    template <typename IndicesT>
    static constexpr void verifyCurrentIndex() {}
  };

  //===--------------------------------------------------------------------===//
  // Argument Values

  /// This class defines an Argument value.
  template <bool isTerminal = false>
  struct ArgValue : public ValueBase<ValueType::kArgument, isTerminal> {
    template <typename IndicesT>
    static constexpr void verifyCurrentIndex() {
      static_assert(isTerminal || IndicesT::argIdx != -1,
                    "Do not place Arguments after RemainingArguments");
      static_assert(
          !isTerminal || IndicesT::argIdx != -1,
          "Do not use more than one RemainingArguments/RepeatedArguments");

      static_assert(IndicesT::resultIdx == 0,
                    "Arguments should appear before results");
      static_assert(IndicesT::attrIdx == 0,
                    "Arguments and results should appear before attributes");
      static_assert(IndicesT::invIdx == 0,
                    "Arguments and results should appear before invokables");
    }
  };
  /// This class defines an Argument value that terminates the argument list,
  /// and consumes the remaining arguments.
  struct TerminalArgValue : public ArgValue</*isTerminal=*/true> {};

  //===--------------------------------------------------------------------===//
  // Result Values

  /// This class represents a Result value.
  template <bool isTerminal = false>
  struct ResultValue : public ValueBase<ValueType::kResult, isTerminal> {
    template <typename IndicesT>
    static constexpr void verifyCurrentIndex() {
      static_assert(isTerminal || IndicesT::resultIdx != -1,
                    "Do not place Results after RemainingResults");
      static_assert(!isTerminal || IndicesT::resultIdx != -1,
                    "Do not use more than one RemainingResults");
      static_assert(IndicesT::attrIdx == 0,
                    "Arguments and results should appear before attributes");
      static_assert(IndicesT::invIdx == 0,
                    "Arguments and results should appear before invokables");
    }
  };
  /// This class defines a Result value that terminates the result list,
  /// and consumes the remaining results.
  struct TerminalResultValue : public ResultValue</*isTerminal=*/true> {};

  //===--------------------------------------------------------------------===//
  // Attribute Values

  /// This class defines an Attribute value.
  template <bool isTerminal = false>
  struct AttrValue : public ValueBase<ValueType::kAttribute, isTerminal> {
    template <typename IndicesT>
    static void verifyCurrentIndex() {
      static_assert(IndicesT::attrIdx != -1,
                    "Do not place Attributes after RemainingMEFAttrs");
      static_assert(!isTerminal || IndicesT::attrIdx != -1,
                    "Do not use more than one RemainingMEFAttrs");
    }
  };

  /// This class defines an Attribute value that terminates the attribute list,
  /// and consumes the remaining attributes.
  struct TerminalAttrValue : public AttrValue</*isTerminal=*/true> {};

  //===--------------------------------------------------------------------===//
  // Invokable Values

  /// This class defines a Invokable value.
  struct InvokableValue : public ValueBase<ValueType::kInvokable> {};

  //===--------------------------------------------------------------------===//
  // Framework Values

  /// This class defines a Framework value, such as a
  /// RunnablePrimitiveDescriptor or Runtime.
  struct FrameworkValue : public ValueBase<ValueType::kFramework> {};

  //===--------------------------------------------------------------------===//
  // Argument handling: Value Definitions
  //===--------------------------------------------------------------------===//

  /// This class defines a concrete type of value, with `T` representing the
  /// expected C++ class. Each specialization should derive from one of the
  /// specific value types above, and also provide a `process` function that
  /// computes the concrete value. This function should generally be of (or
  /// similar to) the following form:
  ///
  ///   template <typename IndicesT>
  ///   static T process(RunnablePrimitiveDescriptor &frame) {
  ///     // Compute and return the value of `T` here.
  ///     // `IndicesT` is an instance of ValueIndices containing the current
  ///     // indices for each value type.
  ///   }
  ///
  /// Note: The `Dummy` type is used to avoid GCC errors for defining struct
  /// specializations outside of a namespace.
  template <typename T, typename Dummy = void>
  struct Value;

  //===--------------------------------------------------------------------===//
  // Argument Values

  /// By default, treat any unknown type as an Argument.
  template <typename T, typename Dummy>
  struct Value : public ArgValue<> {
    using ArgTy = std::decay_t<T>;

    template <typename IndicesT>
    static auto &&process(RunnablePrimitiveDescriptor &frame) {
      return frame.getArg<ArgTy>(IndicesT::argIdx);
    }

    /// Mark if the value is a chain.
    static constexpr bool isChain = std::is_same<ArgTy, Chain>::value;
  };

  /// If this wants an AsyncValueRef, then they want raw low-level access to an
  /// argument, provide it.
  template <typename T, typename Dummy>
  struct Value<AsyncValueRef<T>, Dummy> : public ArgValue<> {
    template <typename IndicesT>
    static AsyncValueRef<T> process(RunnablePrimitiveDescriptor &frame) {
      return std::move(frame.getArg(IndicesT::argIdx));
    }

    /// Mark if the value is a chain.
    static constexpr bool isChain = std::is_same<T, Chain>::value;
  };

  /// If this wants an AsyncValueRef, then they want raw low-level access to an
  /// argument, provide it.
  template <typename Dummy>
  struct Value<AnyAsyncValueRef, Dummy> : public ArgValue<> {
    template <typename IndicesT>
    static AnyAsyncValueRef process(RunnablePrimitiveDescriptor &frame) {
      return std::move(frame.getArg(IndicesT::argIdx));
    }
  };

  /// RemainingArguments provides an ArrayRef containing all remaining
  /// arguments. Useful for variadic primitives with heterogenous elements.
  template <typename Dummy>
  struct Value<RemainingArguments, Dummy> : public TerminalArgValue {
    template <typename IndicesT>
    static RemainingArguments process(RunnablePrimitiveDescriptor &frame) {
      return RemainingArguments(
          frame.getArguments().drop_front(IndicesT::argIdx));
    }
  };

  //===--------------------------------------------------------------------===//
  // Result Values

  /// Result<T> provides an initialized AsyncValue with a specific type.
  template <typename T, typename Dummy>
  struct Value<Result<T>, Dummy> : public ResultValue<> {
    template <typename IndicesT>
    static Result<T> process(RunnablePrimitiveDescriptor &frame) {
      return Result<T>(frame.getResults()[IndicesT::resultIdx], frame);
    }
  };

  /// IndirectResult provides an initialized indirect AsyncValue.
  template <typename Dummy>
  struct Value<IndirectResult, Dummy> : public ResultValue<> {
    template <typename IndicesT>
    static IndirectResult process(RunnablePrimitiveDescriptor &frame) {
      return IndirectResult(frame.getResults()[IndicesT::resultIdx], frame);
    }
  };

  /// RemainingResults provides a view containing all remaining results.
  /// Useful for variadic primitives.
  template <typename Dummy>
  struct Value<RemainingResults, Dummy> : public TerminalResultValue {
    template <typename IndicesT>
    static RemainingResults process(RunnablePrimitiveDescriptor &frame) {
      return RemainingResults(
          frame.getResults().drop_front(IndicesT::resultIdx),
          frame.getRuntime());
    }
  };

  //===--------------------------------------------------------------------===//
  // Attribute Values

  /// Specialization to cast a single attribute (T).
  template <typename T, typename Dummy>
  struct Value<MEFAttr<T>, Dummy> : public AttrValue<> {
    template <typename IndicesT>
    static MEFAttr<T> process(RunnablePrimitiveDescriptor &frame) {
      return frame.getAttr<T>(IndicesT::attrIdx);
    }
  };

  /// RemainingMEFAttrs provides an ArrayRef containing all remaining
  /// attributes. Useful for primitives with unknown dynamic attributes.
  template <typename Dummy>
  struct Value<RemainingMEFAttrs, Dummy> : public TerminalAttrValue {
    template <typename IndicesT>
    static RemainingMEFAttrs process(RunnablePrimitiveDescriptor &frame) {
      return RemainingMEFAttrs(
          frame.getAttrBase(),
          frame.getAttrOffsets().drop_front(IndicesT::attrIdx),
          frame.getAttrSection());
    }
  };

  //===--------------------------------------------------------------------===//
  // Invokable Values

  /// If this takes a `Invokable *`, then pass it down.
  template <typename Dummy>
  struct Value<MEFInvokable *, Dummy> : public InvokableValue {
    template <typename IndicesT>
    static MEFInvokable *process(RunnablePrimitiveDescriptor &frame) {
      return frame.getInvokable(IndicesT::invIdx);
    }
  };

  //===--------------------------------------------------------------------===//
  // Framework Values

  /// If this requires the RunnablePrimitiveDescriptor for any reason, pass it
  /// along.
  template <typename Dummy>
  struct Value<RunnablePrimitiveDescriptor &, Dummy> : public FrameworkValue {
    template <typename IndicesT>
    static RunnablePrimitiveDescriptor &
    process(RunnablePrimitiveDescriptor &frame) {
      return frame;
    }
  };

  /// If this requires a Runtime, pass it along.
  template <typename Dummy>
  struct Value<CPUDevice &, Dummy> : public FrameworkValue {
    template <typename IndicesT>
    static CPUDevice &process(RunnablePrimitiveDescriptor &frame) {
      return frame.getRuntime();
    }
  };

  /// If this requires an EncodedLocation for the frame, pass it along.
  template <typename Dummy>
  struct Value<EncodedLocation, Dummy> : public FrameworkValue {
    template <typename IndicesT>
    static EncodedLocation process(RunnablePrimitiveDescriptor &frame) {
      return frame.copyLocation();
    }
  };

  //===--------------------------------------------------------------------===//
  // Argument handling: Processing
  //===--------------------------------------------------------------------===//

  /// This function handles the nitty gritty processing of the arguments. We
  /// recursively unwrap each argument, verify the current value indices,
  /// process the argument, and append it to the working argument list.
  template <typename IndicesT, bool hasInChain, typename ArgT,
            typename... ArgsToProcessT, typename... ArgsT>
  static void processArgsAndInvoke(TypeTag<ArgT, ArgsToProcessT...>,
                                   RunnablePrimitiveDescriptor &frame,
                                   ArgsT &&...args) {
    Value<ArgT, void>::template verifyCurrentIndex<IndicesT>();
    processArgsAndInvoke<
        typename Value<ArgT, void>::template UpdatedIndicesT<IndicesT>,
        (hasInChain || Value<ArgT, void>::isChain)>(
        TypeTag<ArgsToProcessT...>(), frame, std::forward<ArgsT>(args)...,
        Value<ArgT, void>::template process<IndicesT>(frame));
  }

  /// This function is the base case for argument processing function above. It
  /// is invoked after all arguments have finished processing, and handles some
  /// final validation before invoking the primitive.
  template <typename IndicesT, bool hasInChain, typename... ArgsT>
  static void processArgsAndInvoke(TypeTag<>,
                                   RunnablePrimitiveDescriptor &frame,
                                   ArgsT &&...args) {
    verifyResultIndex<IndicesT::resultIdx>(frame, TypeTag<Return>());

    assert((IndicesT::argIdx == (int)frame.getNumArgs() ||
            IndicesT::argIdx == -1 ||
            (!hasInChain &&
             static_cast<size_t>(IndicesT::argIdx) == frame.getNumArgs() - 1 &&
             frame.getArg(IndicesT::argIdx).template isType<Chain>())) &&
           "Extra arguments passed to primitive");
    assert((IndicesT::attrIdx == (int)frame.getNumAttrs() ||
            IndicesT::attrIdx == -1) &&
           "Extra attributes passed to primitive");
    assert((IndicesT::invIdx == frame.getNumInvokables() ||
            IndicesT::invIdx == -1) &&
           "Extra invokables passed to primitive");
    invoke<Return>(frame, std::forward<ArgsT>(args)...);
  }

  // Verify the result index for non-void Return type
  template <int resultIdx, typename T>
  static void verifyResultIndex(RunnablePrimitiveDescriptor &frame,
                                TypeTag<T>) {
    static_assert(resultIdx == 0,
                  "Don't mix return by value and Result<> syntax.");
  }

  // Verify the result index for void Return type.
  template <int resultIdx>
  static void verifyResultIndex(RunnablePrimitiveDescriptor &frame,
                                TypeTag<void>) {
    assert((resultIdx == (int)frame.getNumResults() || resultIdx == -1) &&
           "Extra results passed to primitive");
  }

  //===--------------------------------------------------------------------===//
  // Result handling
  //===--------------------------------------------------------------------===//

  // Handles the return value of the primitive with `handleReturn` if
  // non-void. Otherwise ignores the return value.
  template <typename ReturnT, typename... ActualArgs>
  static std::enable_if_t<!std::is_same<ReturnT, void>::value>
  invoke(RunnablePrimitiveDescriptor &frame, ActualArgs &&...args) {
    handleReturn(frame, implFn(std::forward<ActualArgs>(args)...));
  }
  template <typename ReturnT, typename... ActualArgs>
  static std::enable_if_t<std::is_same<ReturnT, void>::value>
  invoke(RunnablePrimitiveDescriptor &frame, ActualArgs &&...args) {
    implFn(std::forward<ActualArgs>(args)...);
  }

  // These functions provide custom behavior for concrete result types that
  // may come up.

  /// Store result as an AsyncValue output in RunnablePrimitiveDescriptor by
  /// creating a ConcreteAsyncValue.
  template <typename T>
  static void processResult(RunnablePrimitiveDescriptor &frame, size_t index,
                            T &&t) {
    frame.emplaceResult<std::decay_t<T>>(index, std::forward<T>(t));
  }

  /// Support functions that return ErrorOr<T>: if it contains an error, call
  /// frame.fail() to report the error message and set an error in the output
  /// AsyncValue. Otherwise, store the return value as normal.
  template <typename T>
  static void processResult(RunnablePrimitiveDescriptor &frame, size_t index,
                            ErrorOr<T> &&errorOr) {
    if (succeeded(errorOr))
      processResult(frame, index, errorOr.takeValue());
    else
      frame.fail(errorOr.takeError());
  }

  /// Don't allocate an async value for chain results, just use the one
  /// already stored in Runtime.
  static void processResult(RunnablePrimitiveDescriptor &frame, size_t index,
                            Chain t) {
    frame.setResult(index, frame.getRuntime().getReadyChain().copy());
  }

  /// Handle explicit AnyAsyncValueRef results.
  static void processResult(RunnablePrimitiveDescriptor &frame, size_t index,
                            AnyAsyncValueRef t) {
    frame.setResult(index, std::move(t));
  }

  /// Handle explicit AsyncValueRef results.
  template <typename T>
  static void processResult(RunnablePrimitiveDescriptor &frame, size_t index,
                            AsyncValueRef<T> t) {
    frame.setResult(index, std::move(t));
  }

  /// Store the primitive result back to the output AsyncValue in the
  /// RunnablePrimitiveDescriptor.
  template <typename T>
  static void handleReturn(RunnablePrimitiveDescriptor &frame, T &&t) {
    assert(frame.getNumResults() == 1 && "Incorrect number of results");
    processResult(frame, 0, std::forward<T>(t));
  }

  /// Support implementations that return std::pair<>, storing the two results
  /// in separate output results.
  template <typename T1, typename T2>
  static void handleReturn(RunnablePrimitiveDescriptor &frame,
                           std::pair<T1, T2> &&pair) {
    assert(frame.getNumResults() == 2 && "Incorrect number of results");
    processResult(frame, 0, std::move(pair.first));
    processResult(frame, 1, std::move(pair.second));
  }

  /// Support implementations that return std::tuple<>, storing each element in
  /// separate output results.
  template <typename... Ts>
  static void handleReturn(RunnablePrimitiveDescriptor &frame,
                           std::tuple<Ts...> &&tuple) {
    assert(frame.getNumResults() == sizeof...(Ts) &&
           "Incorrect number of results");

    auto applyFn = [&](auto &&...args) {
      // Process each result in-order.
      int resultIdx = 0;
      (processResult(frame, resultIdx++, std::move(args)), ...);
    };
    std::apply(applyFn, std::move(tuple));
  }
};
} // namespace Detail

} // namespace M

namespace llvm {

template <typename To, typename From>
struct BaseMEFAttrCast
    : public DefaultDoCastIfPossible<std::optional<To>, From,
                                     BaseMEFAttrCast<To, From>> {
  // Accept a const reference in the case of temporaries being casted which get
  // lifetime extension when binding to a const reference.
  static bool isPossible(const From &f) {
    const From &fConst = const_cast<From &>(f);
    auto fUnwrapped = detail::unwrapValue(fConst);

    return To::matchesAttrKind(
        reinterpret_cast<const uint8_t *>(fUnwrapped.get()),
        fUnwrapped.getAttrSection().end());
  }

  // Accept a const reference in the case of temporaries being casted which get
  // lifetime extension when binding to a const reference.
  static std::optional<To> doCast(const From &f) {
    const From &fConst = const_cast<From &>(f);
    auto fUnwrapped = detail::unwrapValue(fConst);
    return To(fUnwrapped.get(), fUnwrapped.getAttrSection());
  }

  static std::optional<To> castFailed() { return std::nullopt; }
};

template <typename To, typename From>
struct BaseMEFAttrCastOptional
    : public DefaultDoCastIfPossible<
          std::optional<To>, std::optional<From>,
          BaseMEFAttrCastOptional<To, std::optional<From>>> {
  // Accept a const reference in the case of temporaries being casted which get
  // lifetime extension when binding to a const reference.
  static bool isPossible(const From &f) {
    const From &fConst = const_cast<From &>(f);
    auto fUnwrapped = detail::unwrapValue(fConst);

    return To::matchesAttrKind(
        reinterpret_cast<const uint8_t *>((*fUnwrapped).get()),
        (*fUnwrapped).getAttrSection().end());
  }

  // Accept a const reference in the case of temporaries being casted which get
  // lifetime extension when binding to a const reference.
  static std::optional<To> doCast(const From &f) {
    const From &fConst = const_cast<From &>(f);
    auto fUnwrapped = detail::unwrapValue(fConst);
    return To((*fUnwrapped).get(), (*fUnwrapped).getAttrSection());
  }

  static std::optional<To> castFailed() { return std::nullopt; }
};

template <typename To>
struct CastInfo<To, ::M::MEFDynamicAttr>
    : public BaseMEFAttrCast<::M::MEFAttr<To>, ::M::MEFDynamicAttr> {};

template <typename To>
struct CastInfo<To, const ::M::MEFDynamicAttr>
    : public BaseMEFAttrCast<::M::MEFAttr<To>, ::M::MEFDynamicAttr> {};

template <typename To>
struct CastInfo<::M::MEFAttr<To>, ::M::MEFDynamicAttr>
    : public BaseMEFAttrCast<::M::MEFAttr<To>, ::M::MEFDynamicAttr> {};

template <typename To>
struct CastInfo<To, std::optional<::M::MEFDynamicAttr>>
    : public BaseMEFAttrCastOptional<::M::MEFAttr<To>, ::M::MEFDynamicAttr> {};

template <typename To>
struct CastInfo<::M::MEFAttr<To>, std::optional<::M::MEFDynamicAttr>>
    : public BaseMEFAttrCastOptional<::M::MEFAttr<To>, ::M::MEFDynamicAttr> {};

template <typename To, typename From>
struct BaseMEFAttrPointerCast
    : public DefaultDoCastIfPossible<std::optional<To>, const From *,
                                     BaseMEFAttrPointerCast<To, From>> {
  static bool isPossible(const From *f) {
    return To::matchesAttrKind(reinterpret_cast<const uint8_t *>(f->get()),
                               f->getAttrSection().end());
  }
  static std::optional<To> doCast(const From *f) {
    return To(f->get(), f->getAttrSection());
  }
  static std::optional<To> castFailed() { return std::nullopt; }
};

template <typename To>
struct CastInfo<To, const ::M::MEFDynamicAttr *>
    : public BaseMEFAttrPointerCast<::M::MEFAttr<To>, ::M::MEFDynamicAttr> {};

template <typename To>
struct CastInfo<::M::MEFAttr<To>, const ::M::MEFDynamicAttr *>
    : public BaseMEFAttrPointerCast<::M::MEFAttr<To>, ::M::MEFDynamicAttr> {};

} // namespace llvm

#endif // MLRT_GRAPHRT_MEFEXECUTOR_PRIMITIVEREGISTRY_H
