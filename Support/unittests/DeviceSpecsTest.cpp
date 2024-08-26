//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DeviceSpecs.h"

#include "gtest/gtest.h"

using namespace M;

namespace {

static DeviceSpecCollection createDeviceSpecCollection() {
  DeviceSpecCollection specs;
  specs.host.label = "cpu";

  {
    DeviceSpec &spec = specs.devices.emplace_back();
    spec.ref.label = "cpu";
    spec.target.triple = llvm::Triple("x86_64-unknown-linux-gnu");
    spec.target.arch = "znver3";
    spec.target.features.emplace_back("avx2");
    spec.target.features.emplace_back("avx");
  }

  {
    DeviceSpec &spec = specs.devices.emplace_back();
    spec.ref.label = "cuda";
    spec.ref.id = 0;
    spec.target.triple = llvm::Triple("nvptx64-nvidia-cuda");
    spec.target.arch = "sm_80";
  }

  {
    DeviceSpec &spec = specs.devices.emplace_back();
    spec.ref.label = "cuda";
    spec.ref.id = 1;
    spec.target.triple = llvm::Triple("nvptx64-nvidia-cuda");
    spec.target.arch = "sm_80";
  }

  return specs;
}

TEST(DevicesSpecs, RoundTrip) {
  DeviceSpecCollection specs = createDeviceSpecCollection();
  ErrorOr<DeviceSpecCollection> roundTrippedOr =
      specs.deserializeFromJSON(specs.serializeToJSON());
  ASSERT_FALSE(roundTrippedOr.isError());
  EXPECT_EQ(roundTrippedOr->serializeToJSON(), specs.serializeToJSON());
}

TEST(DeviceSpecs, FindDeviceSpec) {
  DeviceSpecCollection specs = createDeviceSpecCollection();
  EXPECT_EQ(specs.getHostDeviceSpec().ref.toString(), "cpu:0");
  {
    ErrorOr<const DeviceSpec *> specOr =
        specs.findDeviceSpec(DeviceRef("cuda"));
    ASSERT_FALSE(specOr.isError());
    EXPECT_EQ((*specOr)->ref.toString(), "cuda:0");
  }
  {
    ErrorOr<const DeviceSpec *> specOr =
        specs.findDeviceSpec(DeviceRef("cuda", 1));
    ASSERT_FALSE(specOr.isError());
    EXPECT_EQ((*specOr)->ref.toString(), "cuda:1");
  }
  {
    ErrorOr<const DeviceSpec *> specOr =
        specs.findDeviceSpec(DeviceRef("cuda", 2));
    EXPECT_TRUE(specOr.isError());
  }
}

} // namespace
