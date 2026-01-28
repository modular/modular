//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Tensor manipulation tests
//
//===----------------------------------------------------------------------===//
#ifndef MAX_TESTS_INTERNAL_API_C_UTILS_H
#define MAX_TESTS_INTERNAL_API_C_UTILS_H

#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/device.h"
#include "max/c/model.h"
#include "max/c/tensor.h"
#include "max/c/types.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#define EXPECT_SUCCESS(status, ...)                                            \
  {                                                                            \
    EXPECT_FALSE(M_isError(status))                                            \
        << "" __VA_ARGS__ << ": " << M_getError(status);                       \
  }
#define EXPECT_FAILURE(status, match)                                          \
  {                                                                            \
    EXPECT_TRUE(M_isError(status));                                            \
    EXPECT_THAT(M_getError(status), ::testing::HasSubstr(match));              \
  }

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

namespace M {

class APITest : public ::testing::Test {
protected:
  void SetUp() override {
    status = M_newStatus();
    runtimeConfig = M_newRuntimeConfig();
    context = M_newRuntimeContext(runtimeConfig, status);
    EXPECT_SUCCESS(status, "failed to create runtime");
    host = M_newDevice(M_HOST, 0, status);
    EXPECT_SUCCESS(status, "failed to create host device");
  }

  void TearDown() override {
    M_freeDevice(host);
    M_freeRuntimeContext(context);
    M_freeRuntimeConfig(runtimeConfig);
    M_freeStatus(status);
  }

  M_Status *status;
  M_RuntimeConfig *runtimeConfig;
  M_RuntimeContext *context;
  M_Device *host;
};

class GPUTest : public ::testing::Test {
protected:
  void SetUp() override {
    status = M_newStatus();

    host = M_newDevice(M_HOST, 0, status);
    EXPECT_SUCCESS(status, "failed to create host device");
    int gpuCount = M_getAcceleratorCount();
    ASSERT_GT(gpuCount, 0) << "No GPU available for GPU tests";

    gpuDevice = M_newDevice(M_ACCELERATOR, 0, status);
    ASSERT_FALSE(M_isError(status))
        << "Failed to create GPU device: " << M_getError(status);
    ASSERT_NE(gpuDevice, nullptr);

    host = M_newDevice(M_HOST, 0, status);
    ASSERT_FALSE(M_isError(status)) << "Failed to create host device";
    ASSERT_NE(host, nullptr);

    runtimeConfig = M_newRuntimeConfig();
    M_runtimeConfigAddDevice(runtimeConfig, host);
    M_runtimeConfigAddDevice(runtimeConfig, gpuDevice);

    context = M_newRuntimeContext(runtimeConfig, status);
    EXPECT_SUCCESS(status, "failed to create runtime");
  }

  void TearDown() override {
    M_freeDevice(gpuDevice);
    M_freeDevice(host);
    M_freeRuntimeContext(context);
    M_freeRuntimeConfig(runtimeConfig);
    M_freeStatus(status);
  }

  M_Status *status;
  M_RuntimeConfig *runtimeConfig;
  M_RuntimeContext *context;
  M_Device *gpuDevice = nullptr;
  M_Device *host = nullptr;
};

} // namespace M

#endif
