//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/Invoke.h"
#include "LLCL/Runtime/CompactRuntimePtr.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::LLCL;

TEST(KGENInvokeTest, testInvokeKGENFunction) {
  auto noOp = [](void *, ssize_t, uint8_t, void *, ssize_t, uint8_t) {};

  // Test that we can call a function that returns no arguments.
  KGEN::invoke(noOp, ArrayRef<int32_t>(nullptr, 1),
               ArrayRef<int32_t>(nullptr, 1));

  // Test that we can call the function with a value.
  auto dummyArray = ArrayRef<int32_t>(nullptr, 1);
  KGEN::invoke(noOp, std::forward<decltype(dummyArray)>(dummyArray),
               std::forward<decltype(dummyArray)>(dummyArray));
}

/// Test that we can call the function with a value and get the correct result.
TEST(KGENInvokeTest, testInvokeInterleavedInput) {
  auto addKernel = [](int a, void *, ssize_t, uint8_t, float b) {
    return a + b;
  };
  EXPECT_EQ(KGEN::invoke(addKernel, 1, ArrayRef<float>(nullptr, 1), 2.0f),
            3.0f);
}

/// Can get the correct address for a single input.
TEST(KGENInvokeTest, testInvokeFirstAddress) {
  int32_t array[2] = {1, 2};
  auto getAddr = [](void *ptr, ssize_t, uint8_t) {
    return reinterpret_cast<uintptr_t>(ptr);
  };
  EXPECT_EQ(KGEN::invoke(getAddr, array, std::size(array), DType::si32),
            reinterpret_cast<uintptr_t>(array));
  EXPECT_EQ(KGEN::invoke(getAddr, ArrayRef<int32_t>(array, std::size(array))),
            reinterpret_cast<uintptr_t>(array));
  EXPECT_EQ(KGEN::invoke(getAddr,
                         llvm::MutableArrayRef<int32_t>(array, std::size(array))),
            reinterpret_cast<uintptr_t>(array));
}

/// Can get the correct address for a multiple inputs.
TEST(KGENInvokeTest, testInvokeSecondAddress) {
  int32_t array0[2] = {1, 2}, array1[2] = {3, 4};
  EXPECT_EQ(
      KGEN::invoke([](void *ptr0, ssize_t, uint8_t, void *ptr1, ssize_t,
                      uint8_t) { return reinterpret_cast<uintptr_t>(ptr1); },
                   ArrayRef<int32_t>(array0, std::size(array0)),
                   ArrayRef<int32_t>(array1, std::size(array1))),
      reinterpret_cast<uintptr_t>(array1));
}

TEST(KGENInvokeTest, testInvokeWithTensor) {
  float buffer[1 * 2 * 3];
  Tensor tensor =
      Tensor::createBorrowed(buffer, TensorSpec({1, 2, 3}, DType::f32),
                             /*alignment=*/{}, GML::BufferRef::kLocal);
  EXPECT_EQ(KGEN::invoke([](void *ptr0, ssize_t, ssize_t shape[5],
                            uint8_t) { return shape[1]; },
                         std::forward<Tensor>(tensor)),
            2);

  EXPECT_EQ(KGEN::invoke([](void *ptr0, ssize_t, ssize_t shape[5], uint8_t,
                            size_t val) { return val; },
                         std::forward<Tensor>(tensor), 42),
            42u);
}
