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
// This file implements the 'test' dialect primitives provided by the GraphRT
// library.  These are intended for writing executor testcases.  Notice that
// MANY of these are doing things in weird or non-canonical ways.  This is all
// about unit tests, not about doing things the right way.
//
// See also AsyncRT/unittests/AsyncValueTest.cpp.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "MLRT/GraphRT/Primitives.h"
#include "Support/Threading/Atomics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <thread>

using namespace M;

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

static Chain print_hello(Chain chain) {
  printf("hello mef executor!\n");
  fflush(stdout);
  return chain;
}
static void memory_leak_one_int32(CPUDevice &cpuDevice) {
  // We allocate an integer to intentionally cause a memory leak for testing.
  cpuDevice.getAllocator()->allocate<int32_t>();
}

//===----------------------------------------------------------------------===//
// String, Array, and Dict Tests
//===----------------------------------------------------------------------===//

static std::string get_string(MEFAttr<StringRef> value) { return value->str(); }

static std::string append_string(const std::string &lhs,
                                 const std::string &rhs) {
  return lhs + rhs;
}

static Chain print_string(const std::string &lhs, Chain chain) {
  printf("string = ");
  fwrite(lhs.data(), 1, lhs.size(), stdout);
  fputc('\n', stdout);
  fflush(stdout);
  return chain;
}

static void print_string_attribute(Chain inChain, Result<Chain> outChain,
                                   MEFAttr<StringRef> string) {
  if (string->empty() || string->back() != '\0')
    return std::move(outChain).setToError(
        "expecting string to be null-terminated");
  fwrite(string->data(), 1, string->size() - 1, stdout);
  fflush(stdout);
  std::move(outChain).emplace();
}

static Chain print_array_i32(Chain chain, MEFAttr<ArrayRef<int32_t>> values) {
  printf("array = [");
  for (size_t i = 0, e = values->size(); i != e; ++i) {
    if (i != 0)
      printf(", ");
    printf("%d", values[i]);
  }
  printf("]\n");
  fflush(stdout);
  return chain;
}

static Chain print_array_uint8_hex(Chain chain,
                                   MEFAttr<ArrayRef<uint8_t>> values) {
  printf("array = [");
  for (size_t i = 0, e = values->size(); i != e; ++i) {
    if (i != 0)
      printf(", ");
    printf("0x%02X", values[i]);
  }
  printf("]\n");
  return chain;
}

static ErrorOr<Chain> print_dynamic_array_float(Chain chain,
                                                MEFDynamicAttr value) {
  SmallVector<float> floats;
  if (auto err = value.decode(&floats))
    return err.takeError();

  printf("array = [");
  for (size_t i = 0, e = floats.size(); i != e; ++i) {
    if (i != 0)
      printf(", ");
    printf("%f", floats[i]);
  }
  printf("]\n");
  fflush(stdout);
  return chain;
}

static Chain print_dict_i32(Chain chain, MEFAttr<StringRef> key,
                            MEFDictionaryAttr dict) {
  printf("dictionary[%.*s] = ", (int)key->size(), key->begin());
  auto value = dict.find<int32_t>(*key);
  if (!value)
    printf("NOTFOUND\n");
  else
    printf("%d\n", value->get());

  fflush(stdout);
  return chain;
}

static Chain print_all_dict(Chain chain, MEFDictionaryAttr dict) {
  printf("dictionary = ");
  llvm::interleave(
      dict,
      [](const MEFDictionaryAttr::Element &elt) {
        if (elt.getName() == "aString") {
          auto value = *(elt.getValue<StringRef>());
          printf("aString : %.*s", (int)value.size(), value.data());
        } else {
          printf("%.*s : %d", (int)elt.getName().size(), elt.getName().data(),
                 *(elt.getValue<int32_t>()));
        }
      },
      []() { printf(", "); });

  fputc('\n', stdout);
  fflush(stdout);
  return chain;
}

static void printType(MEFType type) {
  if (auto fnType = dyn_cast<MEFFunctionType>(type)) {
    SmallVector<MEFType> inputs, results;
    fnType->decode(inputs, results);
    printf("(");
    llvm::interleave(
        inputs, [](MEFType type) { printType(type); }, []() { printf(", "); });
    printf(") -> ");
    if (results.size() != 1)
      printf("(");
    llvm::interleave(
        results, [](MEFType type) { printType(type); }, []() { printf(", "); });
    if (results.size() != 1)
      printf(")");
  } else if (auto tensorType = dyn_cast<MEFTensorType>(type)) {
    bool hasRank;
    SmallVector<int64_t> dims;
    MEFType elementType = tensorType->decode(hasRank, dims);
    printf("tensor<");
    if (hasRank) {
      for (int64_t dim : dims) {
        if (dim < 0)
          printf("?x");
        else
          printf("%lldx", (long long)dim);
      }
    } else {
      printf("*x");
    }
    printType(elementType);
    printf(">");
  } else if (auto qType = dyn_cast<MEFQuantizedType>(type)) {
    float scale;
    int64_t zeroPoint;
    auto [storageType, expressedType] = qType->decode(scale, zeroPoint);
    printf("quant<");
    printType(storageType);
    printf(":");
    printType(expressedType);
    printf(", %f:%lld>", (double)scale, (long long)zeroPoint);
  } else if (auto qType = dyn_cast<MEFQuantizedPerAxisType>(type)) {
    SmallVector<float> scales;
    SmallVector<int64_t> zeroPoints;
    uint8_t quantizedDim;
    auto [storageType, expressedType] =
        qType->decode(scales, zeroPoints, quantizedDim);
    printf("quantPerAxis<");
    printType(storageType);
    printf(":");
    printType(expressedType);
    printf(":%d, {", (int)quantizedDim);
    llvm::interleave(
        llvm::zip(scales, zeroPoints),
        [](const auto paramPair) {
          auto [scale, zeroPoint] = paramPair;
          printf("%f:%lld", (double)scale, (long long)zeroPoint);
        },
        []() { printf(", "); });
    printf("}>");
  } else {
    std::string str = type.getKind().getAsString();
    printf("%.*s", (int)str.size(), str.data());
  }
  fflush(stdout);
}

static Chain print_type(Chain chain, MEFAttr<MEFType> type) {
  printf("Type attribute: ");
  printType(type);
  printf("\n");
  fflush(stdout);
  return chain;
}

static Chain print_bool(Chain chain, bool val) {
  printf("bool = %s\n", val ? "true" : "false");
  fflush(stdout);
  return chain;
}

static void printIndent(unsigned indent) {
  for (unsigned i = 0; i != indent; ++i)
    printf("  ");
  fflush(stdout);
}

static void print_attr(MEFDynamicAttr value, unsigned indent) {
  if (auto intAttr = dyn_cast<int32_t>(value)) {
    printf("sint32 = %d\n", intAttr->get());
  } else if (auto intAttr = dyn_cast<uint32_t>(value)) {
    printf("uint32 = %u\n", intAttr->get());
  } else if (auto strAttr = dyn_cast<StringRef>(value)) {
    printf("string = ");
    fwrite(strAttr.value()->data(), 1, strAttr.value()->size(), stdout);
    fputc('\n', stdout);
  } else if (auto intArray = dyn_cast<ArrayRef<int32_t>>(value)) {
    print_array_i32(Chain(), *intArray);
  } else if (auto byteArray = dyn_cast<ArrayRef<uint8_t>>(value)) {
    print_array_uint8_hex(Chain(), *byteArray);
  } else if (auto dict = dyn_cast<MEFDictionaryAttr>(value)) {
    printf("dictionary {\n");
    for (const MEFDictionaryAttr::Element &elt : *dict) {
      printIndent(indent + 1);
      printf("'%.*s': ", (int)elt.getName().size(), elt.getName().data());
      print_attr(elt.getValue(), indent + 1);
    }
    printIndent(indent);
    printf("}\n");
  } else if (auto list = dyn_cast<MEFListAttr>(value)) {
    printf("list [\n");
    for (MEFDynamicAttr elt : *list) {
      printIndent(indent + 1);
      print_attr(elt, indent + 1);
    }
    printIndent(indent);
    printf("]\n");
  } else if (auto type = dyn_cast<MEFAttr<MEFType>>(value)) {
    print_type(Chain(), type.value());
  } else {
    printf("unsupported attribute kind\n");
  }
  fflush(stdout);
}

//===----------------------------------------------------------------------===//
// Dynamic attributes
//===----------------------------------------------------------------------===//

static Chain print_dynamic_attr(Chain chain, MEFDynamicAttr value) {
  print_attr(value, 0);
  return chain;
}

//===----------------------------------------------------------------------===//
// Value Tracking
//===----------------------------------------------------------------------===//

/// This class is just an integer but it prints on construction and destruction
/// so we can write tests showing that the values get created and destroyed at
/// the right times.
class VTValue {
public:
  explicit VTValue(int32_t v) : v(v) {
    printf("constructed vt.value(%d)\n", v);
  }

  VTValue(const VTValue &rhs) : v(rhs.v) {
    printf("copy constructed vt.value(%d)\n", v);
  }

  VTValue(VTValue &&rhs) : v(rhs.v) {
    rhs.v = -1;
    printf("move constructed vt.value(%d)\n", v);
  }

  ~VTValue() {
    if (v == -1)
      return;
    printf("destroyed vt.value(%d)\n", v);
    fflush(stdout);
  }

  int32_t v;
};

static VTValue vt_constant(MEFAttr<int32_t> v) { return VTValue(*v); }

static VTValue vt_add(const VTValue &lhs, const VTValue &rhs) {
  return VTValue(lhs.v + rhs.v);
}

static Chain vt_print(const VTValue &v) {
  printf("print vt.value(%d)\n", v.v);
  return Chain();
}

/// Share input args with results without copying.
static std::pair<AsyncValueRef<VTValue>, AsyncValueRef<VTValue>>
vt_share_to_two(AsyncValueRef<VTValue> in) {
  auto result1 = in.copy();
  return {std::move(result1), std::move(in)};
}

/// Share input args with results without copying.  This is the same as above,
/// but uses AnyAsyncValueRef instead of AsyncValueRef<>.
static std::pair<AnyAsyncValueRef, AnyAsyncValueRef>
vt_share_to_two_2(AnyAsyncValueRef in) {
  auto result1 = in.copy();
  return {std::move(result1), std::move(in)};
}

/// Share input args with results without copying.
static std::tuple<AsyncValueRef<VTValue>, AsyncValueRef<VTValue>,
                  AsyncValueRef<VTValue>, AsyncValueRef<VTValue>>
vt_share_to_four(AsyncValueRef<VTValue> in) {
  auto result1 = in.copy(), result2 = in.copy(), result3 = in.copy();
  return {std::move(result1), std::move(result2), std::move(result3),
          std::move(in)};
}

/// Share input args with results without copying.  This is the same as above,
/// but uses AnyAsyncValueRef instead of AsyncValueRef<>.
static std::tuple<AnyAsyncValueRef, AnyAsyncValueRef, AnyAsyncValueRef,
                  AnyAsyncValueRef>
vt_share_to_four_2(AnyAsyncValueRef in) {
  auto result1 = in.copy(), result2 = in.copy(), result3 = in.copy();
  return {std::move(result1), std::move(result2), std::move(result3),
          std::move(in)};
}

static void registerValueTracking(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "test.vt.constant", vt_constant);
  ADD_PRIMITIVE(registry, "test.vt.add", vt_add);
  ADD_PRIMITIVE(registry, "test.vt.print", vt_print);
  ADD_PRIMITIVE(registry, "test.vt_share_to_two", vt_share_to_two);
  ADD_PRIMITIVE(registry, "test.vt_share_to_two_2", vt_share_to_two_2);
  ADD_PRIMITIVE(registry, "test.vt_share_to_four", vt_share_to_four);
  ADD_PRIMITIVE(registry, "test.vt_share_to_four_2", vt_share_to_four_2);
}

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

// This produces only failure/success as Error.
static ErrorOr<int32_t> fail() { return Error("something bad happened"); }

// This produces a normal output and an error output.
static std::pair<ErrorOr<int32_t>, ErrorOr<int32_t>> partial_fail() {
  return {42, Error("something partially bad happened")};
}

//===----------------------------------------------------------------------===//
// Async Operations
//===----------------------------------------------------------------------===//

static AsyncValueRef<int32_t> async_add_i32(int32_t lhs, int32_t rhs,
                                            CPUDevice &cpuDevice) {
  // Even though a single scalar add is trivial, we can do it on a background
  // thread if we'd like!
  return addTask(cpuDevice, [lhs, rhs] { return lhs + rhs; });
}

/// Async sleep primitive that simulates GPU kernel launch latency.
/// Takes delayNs attribute specifying nanoseconds to sleep on a background
/// thread. This is useful for benchmarking MEF executor scheduling overhead.
static void test_sleep(Chain inChain, Result<Chain> outChain,
                       MEFAttr<int32_t> delayNs, CPUDevice &cpuDevice) {
  addTask(cpuDevice,
          [delayNs = *delayNs, outChain = outChain.copy()]() mutable {
            std::this_thread::sleep_for(std::chrono::nanoseconds(delayNs));
            std::move(outChain).emplace();
          });
}

// It is silly to do this for add of course, but this shows how to handle this.
static void async_add2_i32(AsyncValueRef<int32_t> lhs,
                           AsyncValueRef<int32_t> rhs, Result<int32_t> result) {
  // If this were a non-strict primitive, this could get invoked before lhs/rhs
  // are resolved, and either or both may resolve to an error, so we would need
  // to handle that.
  //
  // This isn't actually declared that way, so this is just for illustration
  // purposes.
  andThenSync(
      std::make_tuple(std::move(lhs), std::move(rhs)),
      [result = result.copy()](AsyncValueRef<int32_t> lhs,
                               AsyncValueRef<int32_t> rhs) mutable {
        // This actually only needs to check for error if this is
        // defined as non-strict.
        if (lhs.isError())
          std::move(result).setToError(lhs.getPointer()->takeDiagnostic());
        else if (rhs.isError())
          std::move(result).setToError(rhs.getPointer()->takeDiagnostic());
        else
          std::move(result).emplace(*lhs + *rhs);
      });
}

/// Example that returns an indirect value with IndirectResult, nominally
/// because this is implemented in async style and returns different types
/// depending on the resolved types of lhs/rhs
static void async_add_any(AnyAsyncValueRef lhs, AnyAsyncValueRef rhs,
                          IndirectResult result) {
  if (lhs.isType<int32_t>() && rhs.isType<int32_t>())
    std::move(result).emplace<int32_t>(lhs.get<int32_t>() + rhs.get<int32_t>());
  else if (lhs.isType<std::string>() && rhs.isType<std::string>())
    std::move(result).emplace<std::string>(lhs.get<std::string>() +
                                           rhs.get<std::string>());
  else
    std::move(result).setToError("invalid operands to async_add_any");
}

/// Example that shows an example that produces an error on a background thread,
/// requiring use of fancy IndirectResult APIs.
static void async_error(IndirectResult result1, IndirectResult result2,
                        EncodedLocation manualLoc, CPUDevice &cpuDevice) {
  // Setting an AsyncValue to an error requires an EncodedLocation, which can be
  // copied/moved into the task manually and painfully as in this example...
  addTask(cpuDevice,
          [result1 = result1.copy(), loc = std::move(manualLoc)]() mutable {
            std::move(result1).setToError(
                {"error generated on a background thread!", std::move(loc)});
          });

  // ... but please don't!  It is more convenient to use the `copyWithLocation`
  // method on [Indirect]Result, which brings it forward for you and has a
  // convenience method to set the error.
  addTask(cpuDevice, [result2 = result2.copyWithLocation()]() mutable {
    std::move(result2).setToError("error generated on a background thread #2!");
  });
}

/// This creates a programmatic number of waiters on an async value, then
/// completes it.
static int32_t test_async_many_waiters(MEFAttr<int32_t> num,
                                       CPUDevice &cpuDevice) {
  auto av = AsyncValueRef<int32_t>::allocate(cpuDevice);
  auto done = AsyncValueRef<Chain>::allocate(cpuDevice);
  std::atomic<int32_t> result = 0;
  /// Add 'num' waiters that just increment result.
  for (int32_t i = 0, e = *num; i != e; ++i)
    av.andThenSync([&result, num = *num, done = done.copy()]() mutable {
      if (++result == num)
        std::move(done).emplace();
    });
  std::move(av).emplace(42);
  await(done);
  return result;
}

/// This creates a programmatic # tasks as stress test.  This is not written
/// in an idiomatic style, do not follow this pattern!
static void test_many_tasks(Result<int32_t> result, MEFAttr<int32_t> num,
                            CPUDevice &cpuDevice) {
  if (*num == 0)
    return std::move(result).setToError("zero not supported");
  std::atomic<int32_t> counter(*num);

  // Compiler optimization barrier.
  volatile auto numVal = *num;

  // Add 'num' tasks that just decrements the atomic, which then completes the
  // AV.  This is really horrible, never do something like this in your code.
  for (int32_t i = 0, e = *num; i != e; ++i) {
    addTask(cpuDevice,
            // Capture a bunch of stuff so we exceed the inline size of lambdas.
            [numVal, &counter, result = result.copy(), a = int64_t(numVal + 1),
             b = int64_t(numVal * 2), c = int64_t(numVal * 42), r = &cpuDevice,
             s = &cpuDevice + 1, t = &cpuDevice + 2]() mutable {
              // Note: tasks shouldn't sleep or block!
              std::this_thread::sleep_for(std::chrono::milliseconds(1));

              // Make all the captures look used.
              if (a < numVal || b < numVal || c < numVal) {
                // Note: this should never print.  If it does, it means that
                // the closure was corrupted.
                printf("%p, %p, %p %d %d %d %d\n", (void *)r, (void *)s,
                       (void *)t, (int)a, (int)b, (int)c, (int)numVal);
              }

              if (--counter == 0)
                std::move(result).emplace(numVal);
            });
  }

  // Block for all the tasks to finish.  (bad!)
  await(result.copy());
}

/// This test is a nasty test that tries to cause havoc when a thread pool is
/// being torn down, by creating tasks that take a long time to run (thus
/// allowing the test driver to get to the point where it is tearing down the
/// cpuDevice), and then add another task to the thread after workqueue teardown
/// has started.
static void test_crash_threadpool_shutdown(CPUDevice &cpuDevice) {
  addTask(cpuDevice, [&]() {
    for (int i = 0; i < 4; ++i) {
      addTask(cpuDevice, [&cpuDevice, i]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        printf("task %d\n", i);
        addTask(cpuDevice, [&cpuDevice, i]() {
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
          printf("task %d middle\n", i);
          addTask(cpuDevice, [i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            printf("task %d shutdown\n", i);
          });
        });
      });
    }
  });
}

//===----------------------------------------------------------------------===//
// TestVector Primitives
//===----------------------------------------------------------------------===//

// This is intended to test parallel operations and other "advanced"
// use-cases.
namespace {
/// This class is a designed-to-be-simple data type that we can pass around to
/// illustrate some more advanced usecases for async value and related types.
struct TestVector {
  using ValueType = int32_t;

  TestVector() = default;
  TestVector(const TestVector &) = default;
  TestVector(size_t n) { values.resize(n); }
  SmallVector<ValueType> values;
};
/// This class is used for examples that require float32 values.
struct TestFloat32Vector {
  using ValueType = float;

  TestFloat32Vector() = default;
  TestFloat32Vector(const TestFloat32Vector &) = default;
  TestFloat32Vector(size_t n) { values.resize(n); }
  SmallVector<ValueType> values;
};
} // namespace

static TestVector vec_iota(MEFAttr<int32_t> number) {
  TestVector result;
  result.values.reserve(*number);
  for (int32_t i = 0, e = *number; i != e; ++i)
    result.values.push_back(i);
  return result;
}

static TestFloat32Vector vec_float32_iota(MEFAttr<int32_t> size,
                                          MEFAttr<float> step) {
  TestFloat32Vector result;
  result.values.reserve(*size);
  for (int32_t i = 0, e = *size; i != e; ++i)
    result.values.push_back(*step * i);
  return result;
}

static TestVector vec_constant(MEFAttr<int32_t> size, MEFAttr<int32_t> value) {
  TestVector result;
  result.values.reserve(*size);
  for (int32_t i = 0, e = *size; i != e; ++i)
    result.values.push_back(*value);
  return result;
}

static TestFloat32Vector vec_float32_constant(MEFAttr<int32_t> size,
                                              MEFAttr<float> value) {
  TestFloat32Vector result;
  result.values.reserve(*size);
  for (int32_t i = 0, e = *size; i != e; ++i)
    result.values.push_back(*value);
  return result;
}

static ErrorOr<TestVector> vec_add(const TestVector &lhs,
                                   const TestVector &rhs) {
  if (lhs.values.size() != rhs.values.size())
    return Error("vector length mismatch");

  TestVector result;
  result.values.reserve(lhs.values.size());
  for (int32_t i = 0, e = lhs.values.size(); i != e; ++i)
    result.values.push_back(lhs.values[i] + rhs.values[i]);
  return result;
}

/// This version of vector addition does the entire add of the data on one
/// background task.
static void vec_add_async(AsyncValueRef<TestVector> lhs,
                          AsyncValueRef<TestVector> rhs,
                          Result<TestVector> result, CPUDevice &cpuDevice) {
  if (lhs->values.size() != rhs->values.size())
    return std::move(result).setToError("vector length mismatch");

  addTask(cpuDevice, [lhs = std::move(lhs), rhs = std::move(rhs),
                      result = result.copy()]() mutable {
    auto &lhsValues = lhs->values, &rhsValues = rhs->values;
    TestVector resultValues;
    for (int32_t i = 0, e = lhsValues.size(); i != e; ++i)
      resultValues.values.push_back(lhsValues[i] + rhsValues[i]);
    // After we've populated the vector, set it into the result value.
    std::move(result).emplace(std::move(resultValues));
  });
}

/// This version of vector addition does the addition with each element in
/// parallel as separate tasks.
///
/// This is similar to vec_add_async, but uses `parallelForEachNCompletion`.
static void vec_add_async2(AsyncValueRef<TestVector> lhs,
                           AsyncValueRef<TestVector> rhs,
                           Result<TestVector> result, CPUDevice &cpuDevice) {
  size_t lhsValueSize = lhs->values.size();
  if (lhsValueSize != rhs->values.size())
    return std::move(result).setToError("vector length mismatch");

  // Process each element in parallel.
  parallelForEachNFinishing(
      cpuDevice, lhsValueSize, TestVector(lhsValueSize), result.copy(),
      [lhs = std::move(lhs), rhs = std::move(rhs)](size_t n,
                                                   TestVector &resultVec) {
        resultVec.values[n] = lhs->values[n] + rhs->values[n];
      });
}

/// Accumulates the values in the input vector and produces a scalar.
static TestVector::ValueType vec_accumulate(const TestVector &vec) {
  return std::accumulate(vec.values.begin(), vec.values.end(), 0);
}

/// This version of vector accumulation does the entire accumulate of the vector
/// using one background task.
static void vec_accumulate_async(AsyncValueRef<TestVector> vec,
                                 Result<TestVector::ValueType> result,
                                 CPUDevice &cpuDevice) {
  addTask(cpuDevice, [vec = std::move(vec), result = result.copy()]() mutable {
    auto accum = std::accumulate(vec->values.begin(), vec->values.end(), 0);
    // After we've computed the sum, set it into the result value.
    std::move(result).emplace(accum);
  });
}

/// This version of vector accumulate does the accumulation with each batch in
/// parallel as separate tasks.
///
/// This version uses atomics to perform the global accumulation.
static void vec_accumulate_async2(AsyncValueRef<TestVector> vec,
                                  Result<TestVector::ValueType> result,
                                  CPUDevice &cpuDevice) {
  // Get the number of elements in the input vector
  size_t vecLen = vec->values.size();
  // Get the number of tasks to spawn in parallel from the cpuDevice
  //
  // NOTE: In practice you want to use a batch size that is a multiple of the
  // of the size of the workqueue. This would remove blocking on a big/little
  // system
  size_t numBatches = cpuDevice.getWorkQueue()->getParallelismLevel();
  // The batchSize is the number of elements each thread will process
  // we round up the division so that the numBatches*batchSize
  // is at least equal to vecLen.
  //
  // For example, if the vecLen = 100, numBatches=48, then we want
  // the batchSize = 3
  size_t batchSize = (vecLen + numBatches - 1) / numBatches;

  // Process each batch in parallel. The completion function will be called
  // when all threads have completed.
  parallelForEachNCustomCompletion(
      cpuDevice, numBatches,
      [vec = std::move(vec),
       batchSize](size_t n, std::atomic<TestVector::ValueType> &accumResult) {
        ArrayRef<TestVector::ValueType> vecValues = vec->values;
        size_t startPos = n * batchSize;
        // If we exceeded the range of the vector, then there is nothing to do
        if (startPos >= vecValues.size())
          return;
        // The end position is either the end position of the current batch, or
        // the end of the vector whichever is smaller
        size_t endPos = std::min(vecValues.size(), (n + 1) * batchSize);
        // Accumulate the vector values in the range [startPos, endPos), then
        // add it to the global accumResult atomic
        accumResult += std::accumulate(vecValues.begin() + startPos,
                                       vecValues.begin() + endPos, 0);
      },
      // The completion function is called by a single thread when all prior
      // tasks are complete
      [result = result.copy()](
          std::atomic<TestVector::ValueType> &accumResult) mutable {
        // marks the result asyncvalue ready when all the subtasks are
        // completed.
        std::move(result).emplace(accumResult.load());
      },
      // pass the initial accum to each function. It will be converted to an
      // atomic in the parallelForEachNCustomCompletion function
      0);
}

/// A parallel [Softmax](https://en.wikipedia.org/wiki/Softmax_function)
/// implementation
///
/// The Softmax of a given a vector $V = \left [ a_{0}, a_{1}, \cdots, a_{n}
/// \right ]$ is defined by $\displaystyle \frac{e^{a_i}}{\sum_{i =0}^n
/// e^{a_i}}$. For numerical stability, however, we define Softmax as
/// $\displaystyle \frac{e^{a_i - max(V)}}{\sum_{i =0}^n e^{a_i - max(V)}}$.
///
/// The implementation pseudocode is:
///
/// def softmax(Input):
///   N = len(Input)
///   Output = [0]*N
///   maxVal = -inf
///   denom = 0
///   # STAGE 1: Find the max value in the input vector
///   for i in range(N):
///     maxVal = max(maxVal, Input[i])
///   # STAGE 2: Compute the exponential and denominator
///   for i in range(N):
///     Output[i] = exp(Input[i] - maxVal)
///     denom += Output[i]
///   # STAGE 3: Normalize the output
///   for i in range(N):
///     Output[i] /= denom
///   return Output
///
/// The implementation uses the same ideas in vec_accumulate_async2.
static void vec_softmax_async(AsyncValueRef<TestFloat32Vector> input,
                              Result<TestFloat32Vector> output,
                              CPUDevice &cpuDevice) {
  using ElementType = TestFloat32Vector::ValueType;

  // Get the number of elements in the input vector
  size_t vecLen = input->values.size();
  // The batchSize is the number of elements each work item will process.We want
  // the number of batchSize to be large enough to fully utilize the processor's
  // capabilities.
  size_t batchSize = 1024;
  // The number of tasks to spawn in parallel from the cpuDevice.
  // We round up the division so that the numBatches*batchSize
  // is at least equal to vecLen.
  size_t numBatches = (vecLen + batchSize - 1) / batchSize;

  auto maxVal = AnyAsyncValueRef::allocate<ElementType>(cpuDevice);

  // STAGE 1: Find the max value in the input vector
  parallelForEachNCustomCompletion(
      cpuDevice, numBatches,
      [input = input.copy(), vecLen,
       batchSize](size_t n, std::atomic<ElementType> &maxValueResult) {
        ArrayRef<ElementType> inputValues = input->values;
        size_t startPos = n * batchSize;
        // If we exceeded the range of the vector, then there is nothing to do
        if (startPos >= vecLen)
          return;
        // The end position is either the end position of the current batch, or
        // the end of the vector whichever is smaller
        size_t endPos = std::min(vecLen, (n + 1) * batchSize);
        // Find the max element of the vector values in the range [startPos,
        // endPos), then add it to the global accumResult atomic
        atomicMax(maxValueResult,
                  *std::max_element(inputValues.begin() + startPos,
                                    inputValues.begin() + endPos));
      },
      // The completion function is called by a single thread when all prior
      // tasks are complete
      [maxVal =
           maxVal.copy()](std::atomic<ElementType> &maxValueResult) mutable {
        // marks the result asyncvalue ready when all the subtasks are
        // completed.
        std::move(maxVal).emplace<ElementType>(maxValueResult);
      },
      // pass the initial value to each function. It will be converted to an
      // atomic in the parallelForEachNCustomCompletion function
      std::numeric_limits<float>::lowest());

  // The next phase computes the denominator as well as a copy of the vector
  // data with all the elements passed through `exp(x)` function.
  struct DenomAndExpVector {
    TestFloat32Vector expElements;
    ElementType denom;
  };

  auto denomAndExpVectorAV =
      AsyncValueRef<DenomAndExpVector>::allocate(cpuDevice);

  // STAGE 2: Compute the exponential and denominator.
  // STAGE 2 is only executed when maxVal is ready (i.e. STAGE 1 is complete)
  //
  // Note: we do not mark the output as ready until STAGE 3 is complete.
  std::move(maxVal).andThenSync(
      [input = input.copy(), denomAndExpVectorAV = denomAndExpVectorAV.copy(),
       batchSize, &cpuDevice, vecLen,
       numBatches](AnyAsyncValueRef &&maxVal) mutable {
        parallelForEachNCustomCompletion(
            cpuDevice, numBatches,
            [input = std::move(input), maxVal = maxVal.get<ElementType>(),
             vecLen, batchSize](size_t n, TestFloat32Vector &expElements,
                                std::atomic<ElementType> &denom) {
              ArrayRef<ElementType> inputValues = input->values;
              size_t startPos = n * batchSize;
              if (startPos >= vecLen)
                return;
              size_t endPos = std::min(vecLen, (n + 1) * batchSize);
              ElementType denomAccumLocal = 0;
              for (size_t i = startPos; i < endPos; ++i) {
                ElementType val = std::exp(inputValues[i] - maxVal);
                expElements.values[i] = val;
                denomAccumLocal += val; // locally accumulate the denominator
              }
              // atomic add the floating point values.
              atomicAdd(denom, denomAccumLocal);
            },
            [resultAV = denomAndExpVectorAV.copy()](
                TestFloat32Vector &expElements,
                std::atomic<ElementType> &denom) mutable {
              // store the denomValue in the denom asyncvalue and mark as ready
              std::move(resultAV).emplace(
                  DenomAndExpVector{std::move(expElements), denom.load()});
            },
            vecLen, 0);
      });

  // STAGE 3: Normalize the output
  // STAGE 3 is only executed when denom is ready (i.e. STAGE 2 is complete)
  std::move(denomAndExpVectorAV)
      .andThenSync(
          [output = output.copy(), batchSize, &cpuDevice, vecLen,
           numBatches](AsyncValueRef<DenomAndExpVector> &&denomAndExpVector) {
            // compute STAGE 3 in parallel and mark the output is ready when all
            // work is done
            parallelForEachNFinishing(
                cpuDevice, numBatches,
                std::move(denomAndExpVector->expElements), output.copy(),
                [vecLen, batchSize, denom = denomAndExpVector->denom](
                    size_t n, TestFloat32Vector &expElements) {
                  size_t startPos = n * batchSize;
                  if (startPos >= vecLen)
                    return;
                  size_t endPos = std::min(vecLen, (n + 1) * batchSize);
                  for (size_t i = startPos; i < endPos; ++i)
                    expElements.values[i] /= denom;
                });
          });
}

/// Print an integer vector to the console.
static Chain vec_print(const TestVector &vec, Chain chain) {
  printf("vec = ");
  for (size_t i = 0, e = vec.values.size(); i != e; ++i) {
    if (i != 0)
      printf(", ");
    printf("%d", vec.values[i]);
  }
  fputc('\n', stdout);
  return chain;
}

/// Print a floating point vector to the console.
static Chain vec_float32_print(const TestFloat32Vector &vec, Chain chain) {
  printf("vec.f32 = ");
  for (size_t i = 0, e = vec.values.size(); i != e; ++i) {
    if (i != 0)
      printf(", ");
    printf("%f", vec.values[i]);
  }
  fputc('\n', stdout);
  return chain;
}

static void registerTestVecPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "test.vec.iota", vec_iota);
  ADD_PRIMITIVE(registry, "test.vec.constant", vec_constant);
  ADD_PRIMITIVE(registry, "test.vec.add", vec_add);
  ADD_PRIMITIVE(registry, "test.vec.add.async", vec_add_async);
  ADD_PRIMITIVE(registry, "test.vec.add.async2", vec_add_async2);
  ADD_PRIMITIVE(registry, "test.vec.accumulate", vec_accumulate);
  ADD_PRIMITIVE(registry, "test.vec.accumulate.async", vec_accumulate_async);
  ADD_PRIMITIVE(registry, "test.vec.accumulate.async2", vec_accumulate_async2);
  ADD_PRIMITIVE(registry, "test.vec.print", vec_print);

  // Primitives that operate on floating point vectors.
  ADD_PRIMITIVE(registry, "test.vec.f32.iota", vec_float32_iota);
  ADD_PRIMITIVE(registry, "test.vec.f32.constant", vec_float32_constant);
  ADD_PRIMITIVE(registry, "test.vec.f32.softmax.async", vec_softmax_async);
  ADD_PRIMITIVE(registry, "test.vec.f32.print", vec_float32_print);
}

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

void grt_testcase() {}

AnyAsyncValueRef grt_debug_print_num_refs(AnyAsyncValueRef input) {
  llvm::outs() << "operand has "
               << input.getPointer()->getRefCountForDebugging() << " refs\n";
  llvm::outs().flush();
  return input;
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// This registers the 'test' dialect primitives, intended for writing
/// executor testcases.
void M::registerGraphRTTestPrimitives(PrimitiveRegistry &registry) {
  // grt.testcase
  ADD_PRIMITIVE(registry, "grt.testcase", grt_testcase);

  // grt.debug.assert_unique
  ADD_PRIMITIVE(registry, "grt.debug.print_num_refs", grt_debug_print_num_refs);

  // Misc.
  ADD_PRIMITIVE(registry, "test.print_hello", print_hello);
  ADD_PRIMITIVE(registry, "test.memory_leak_one_int32", memory_leak_one_int32);

  // String and Array Tests
  ADD_PRIMITIVE(registry, "test.get_string", get_string);
  ADD_PRIMITIVE(registry, "test.append_string", append_string);
  ADD_PRIMITIVE(registry, "test.print_string", print_string);
  ADD_PRIMITIVE(registry, "test.print_string_attribute",
                print_string_attribute);
  ADD_PRIMITIVE(registry, "test.print_array_i32", print_array_i32);
  ADD_PRIMITIVE(registry, "test.print_dynamic_array_float",
                print_dynamic_array_float);
  ADD_PRIMITIVE(registry, "test.print_dict_i32", print_dict_i32);
  ADD_PRIMITIVE(registry, "test.print_all_dict", print_all_dict);
  ADD_PRIMITIVE(registry, "test.print_type", print_type);

  ADD_PRIMITIVE(registry, "test.print_bool", print_bool);

  ADD_PRIMITIVE(registry, "test.print.dynamic.attr", print_dynamic_attr);

  registerValueTracking(registry);
  registerTestVecPrimitives(registry);

  // Error handling.
  ADD_PRIMITIVE(registry, "test.fail", fail);
  ADD_PRIMITIVE(registry, "test.partial_fail", partial_fail);

  ADD_PRIMITIVE(registry, "test.async.add.i32", async_add_i32);
  ADD_PRIMITIVE(registry, "test.async.add2.i32", async_add2_i32);
  ADD_PRIMITIVE(registry, "test.async.add.any", async_add_any);
  ADD_PRIMITIVE(registry, "test.async.error", async_error);
  ADD_PRIMITIVE(registry, "test.async.many.waiters", test_async_many_waiters);
  ADD_PRIMITIVE(registry, "test.many.tasks", test_many_tasks);
  ADD_PRIMITIVE(registry, "test.crash.threadpool", test_many_tasks);
  ADD_PRIMITIVE(registry, "test.crash.threadpool.shutdown",
                test_crash_threadpool_shutdown);

  // Benchmark utilities
  ADD_PRIMITIVE(registry, "test.sleep", test_sleep);
}
