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

TEST(KGENInvokeTest, testinvokeKGENFunction) {
  auto noOp = [](void *, ssize_t, uint8_t, void *, ssize_t, uint8_t) {};

  // Test that we can call a function that returns no arguments.
  KGEN::invoke(noOp, llvm::makeArrayRef<int32_t>(nullptr, 1),
               llvm::makeArrayRef<int32_t>(nullptr, 1));

  // Test that we can call the function with a value.
  auto dummyArray = llvm::makeArrayRef<int32_t>(nullptr, 1);
  KGEN::invoke(noOp, std::forward<decltype(dummyArray)>(dummyArray),
               std::forward<decltype(dummyArray)>(dummyArray));
}

/// Test that we can call the function with a value and get the correct result.
TEST(KGENInvokeTest, testinvokeInterleavedInput) {
  auto addKernel = [](int a, void *, ssize_t, uint8_t, float b) {
    return a + b;
  };
  EXPECT_EQ(
      KGEN::invoke(addKernel, 1, llvm::makeArrayRef<float>(nullptr, 1), 2.0f),
      3.0f);
}

/// Can get the correct address for a single input.
TEST(KGENInvokeTest, testinvokeFirstAddress) {
  int32_t arry[2] = {1, 2};
  auto getAddr = [](void *ptr, ssize_t, uint8_t) {
    return reinterpret_cast<uintptr_t>(ptr);
  };
  EXPECT_EQ(KGEN::invoke(getAddr, arry, std::size(arry), DType::si32),
            reinterpret_cast<uintptr_t>(arry));
  EXPECT_EQ(
      KGEN::invoke(getAddr, llvm::makeArrayRef<int32_t>(arry, std::size(arry))),
      reinterpret_cast<uintptr_t>(arry));
  EXPECT_EQ(KGEN::invoke(getAddr, llvm::makeMutableArrayRef<int32_t>(
                                      arry, std::size(arry))),
            reinterpret_cast<uintptr_t>(arry));
}

/// Can get the correct address for a multiple inputs.
TEST(KGENInvokeTest, testinvokeSecondAddress) {
  int32_t arry0[2] = {1, 2}, arry1[2] = {3, 4};
  EXPECT_EQ(
      KGEN::invoke([](void *ptr0, ssize_t, uint8_t, void *ptr1, ssize_t,
                      uint8_t) { return reinterpret_cast<uintptr_t>(ptr1); },
                   llvm::makeArrayRef<int32_t>(arry0, std::size(arry0)),
                   llvm::makeArrayRef<int32_t>(arry1, std::size(arry1))),
      reinterpret_cast<uintptr_t>(arry1));
}

TEST(KGENInvokeTest, testinvokeWithTensor) {
  Tensor tensor =
      Tensor::createBorrowed(nullptr, TensorSpec({1, 2, 3}, DType::f32),
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
