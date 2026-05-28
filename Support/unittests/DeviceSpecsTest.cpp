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

TEST(DeviceSpecs, DisabledFeaturesRoundTrip) {
  DeviceSpec spec;
  spec.ref.label = "cpu";
  spec.target.triple = llvm::Triple("x86_64-unknown-linux-gnu");
  spec.target.arch = "znver3";
  spec.target.features = {"avx2", "bmi1"};
  spec.target.disabledFeatures = {"avx512f"};

  ErrorOr<DeviceSpec> roundTrippedOr =
      DeviceSpec::deserializeFromJSON(spec.serializeToJSON());
  ASSERT_FALSE(roundTrippedOr.isError());
  EXPECT_EQ(roundTrippedOr->target.features, spec.target.features);
  EXPECT_EQ(roundTrippedOr->target.disabledFeatures,
            spec.target.disabledFeatures);
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

TEST(DeviceSpecs, EncodeFeaturesUnsigned) {
  TargetInfo ti(llvm::Triple(""), "", {"avx2", "bmi1"});
  EXPECT_EQ(encodeFeatures(ti), "+avx2,+bmi1");
}

TEST(DeviceSpecs, EncodeFeaturesWithDisabled) {
  TargetInfo ti(llvm::Triple(""), "", {"avx2", "bmi1"}, {"avx512f"});
  EXPECT_EQ(encodeFeatures(ti), "+avx2,+bmi1,-avx512f");
}

TEST(DeviceSpecs, DecodeFeaturesPositive) {
  ErrorOr<DecodedFeatures> result = decodeFeatures("+avx2,+bmi1");
  ASSERT_FALSE(result.isError());
  EXPECT_EQ(result->enabled, (std::vector<std::string>{"avx2", "bmi1"}));
  EXPECT_TRUE(result->disabled.empty());
}

TEST(DeviceSpecs, DecodeFeaturesNegative) {
  ErrorOr<DecodedFeatures> result =
      decodeFeatures("+avx2,+bmi1,-avx512f,-avx512bw");
  ASSERT_FALSE(result.isError());
  EXPECT_EQ(result->enabled, (std::vector<std::string>{"avx2", "bmi1"}));
  EXPECT_EQ(result->disabled,
            (std::vector<std::string>{"avx512f", "avx512bw"}));
}

TEST(DeviceSpecs, EncodeDecodeRoundTrip) {
  TargetInfo ti(llvm::Triple(""), "", {"avx2", "bmi1"}, {"avx512f"});
  std::string encoded = encodeFeatures(ti);
  EXPECT_EQ(encoded, "+avx2,+bmi1,-avx512f");
  ErrorOr<DecodedFeatures> decoded = decodeFeatures(encoded);
  ASSERT_FALSE(decoded.isError());
  EXPECT_EQ(decoded->enabled, ti.features);
  EXPECT_EQ(decoded->disabled, ti.disabledFeatures);
}

TEST(DeviceSpecs, DecodeFeaturesNormalizesUnsigned) {
  // Unsigned names are treated as enabled for backward compat with older
  // serialized TargetInfos that predate the signed format.
  ErrorOr<DecodedFeatures> result = decodeFeatures("avx2,bmi1");
  ASSERT_FALSE(result.isError());
  EXPECT_EQ(result->enabled, (std::vector<std::string>{"avx2", "bmi1"}));
  EXPECT_TRUE(result->disabled.empty());

  result = decodeFeatures("+avx2,bmi1");
  ASSERT_FALSE(result.isError());
  EXPECT_EQ(result->enabled, (std::vector<std::string>{"avx2", "bmi1"}));
}

} // namespace
