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
    spec.target.cpu = "znver3";
    spec.target.features.emplace_back("avx2");
    spec.target.features.emplace_back("avx");
  }

  {
    DeviceSpec &spec = specs.devices.emplace_back();
    spec.ref.label = "cuda";
    spec.ref.id = 0;
    spec.target.triple = llvm::Triple("nvptx64-nvidia-cuda");
    spec.target.cpu = "compute_80";
  }

  {
    DeviceSpec &spec = specs.devices.emplace_back();
    spec.ref.label = "cuda";
    spec.ref.id = 1;
    spec.target.triple = llvm::Triple("nvptx64-nvidia-cuda");
    spec.target.cpu = "compute_80";
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

TEST(DeviceSpecs, ReconcileDeviceSpecs) {
  DeviceSpecCollection required = createDeviceSpecCollection();
  DeviceSpecCollection provided = createDeviceSpecCollection();

  // Change the labels and ids of the provided.
  provided.host.label = "cpux";
  provided.devices[0].ref.label = "cpux";
  provided.devices[0].target.features.emplace_back("avx512");
  provided.devices[1].ref.label = "cudax";
  provided.devices[1].ref.id = 3;
  provided.devices[2].ref.label = "cudax";
  provided.devices[2].ref.id = 7;

  // The map should be from required refs to provided specs.
  ErrorOr<DeviceSpecMap> mapOr = provided.reconcileDeviceSpecs(required);
  ASSERT_FALSE(mapOr.isError());

  {
    auto itr = mapOr->find(DeviceRef("cpu"));
    ASSERT_TRUE(itr != mapOr->end());
    ASSERT_EQ(itr->first.label, "cpu");
  }

  {
    auto itr = mapOr->find(DeviceRef("cuda"));
    ASSERT_TRUE(itr != mapOr->end());
    ASSERT_EQ(itr->second.first.ref.label, "cuda");
    ASSERT_EQ(itr->second.first.ref.id, 0);
    ASSERT_EQ(itr->second.second.ref.label, "cudax");
    ASSERT_EQ(itr->second.second.ref.id, 3);
  }

  {
    auto itr = mapOr->find(DeviceRef("cuda", 1));
    ASSERT_TRUE(itr != mapOr->end());
    ASSERT_EQ(itr->second.first.ref.label, "cuda");
    ASSERT_EQ(itr->second.first.ref.id, 1);
    ASSERT_EQ(itr->second.second.ref.label, "cudax");
    ASSERT_EQ(itr->second.second.ref.id, 7);
  }
}

} // namespace
