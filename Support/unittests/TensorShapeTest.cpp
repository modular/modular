//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/TensorSpec.h"

#include "Support/ErrorOr.h"

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/YAMLTraits.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace M;

namespace {

static_assert(std::is_same_v<std::decay_t<decltype(mlir::ShapedType::kDynamic)>,
                             std::int64_t>,
              "This file assumes kDynamic is an int64_t");
constexpr std::int64_t kDynamic = mlir::ShapedType::kDynamic;

TensorShape shape(const std::vector<std::int64_t> &vec) {
  return TensorShape(vec);
}

template <typename T>
ErrorOr<T> ok(T v) {
  return std::move(v);
}

bool tensorShapeRoundTrips(const std::vector<std::int64_t> &vec) {
  TensorShape shape(vec);
  std::vector<std::int64_t> round_tripped;
  for (auto dim : shape)
    round_tripped.push_back(dim);
  return vec == round_tripped;
}

struct ShapeHolder {
  TensorShape shape;
};

} // namespace

template <>
struct llvm::yaml::MappingTraits<ShapeHolder> {
  static void mapping(llvm::yaml::IO &io, ShapeHolder &holder) {
    io.mapRequired("shape", holder.shape);
  }
};

TEST(TensorShape, constructor) {
  {
    SmallVector<int64_t> shape{1, 2, 3, 4};
    EXPECT_EQ(TensorShape(shape).getAsString(), "1x2x3x4");
  }

  {
    SmallVector<size_t> shape{1, 2, 3, 4};
    EXPECT_EQ(TensorShape(shape).getAsString(), "1x2x3x4");
  }

  {
    SmallVector<ssize_t> shape{1, 2, 3, 4};
    EXPECT_EQ(TensorShape(shape).getAsString(), "1x2x3x4");
  }

  EXPECT_EQ(TensorShape({1, 2, 3, 4}).getAsString(), "1x2x3x4");

  {
    TensorShape shape = {1, 2, 3, 4};
    EXPECT_EQ(shape.getAsString(), "1x2x3x4");
  }

  {
    TensorShape shape = {1, 2, 3, 4};
    EXPECT_EQ(TensorShape(shape.getDims()), shape);
  }
}

TEST(TensorShape, representations) {
  EXPECT_TRUE(tensorShapeRoundTrips({}));
  EXPECT_TRUE(tensorShapeRoundTrips({1}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, 1}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, 1, 1}));
  EXPECT_TRUE(tensorShapeRoundTrips({100000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 100000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 100000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, 100000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, 1, 100000}));
  EXPECT_TRUE(tensorShapeRoundTrips({10000000000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 10000000000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 10000000000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, 10000000000}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, 1, 10000000000}));
  EXPECT_TRUE(tensorShapeRoundTrips({kDynamic}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, kDynamic}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, kDynamic}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, kDynamic}));
  EXPECT_TRUE(tensorShapeRoundTrips({1, 1, 1, 1, kDynamic}));
}

TEST(TensorShape, stringizing) {
  EXPECT_EQ("", shape({}).getAsString());
  EXPECT_EQ("5", shape({5}).getAsString());
  EXPECT_EQ("5x10", shape({5, 10}).getAsString());
  EXPECT_EQ("5x10x20", shape({5, 10, 20}).getAsString());
  EXPECT_EQ("?x10x20", shape({kDynamic, 10, 20}).getAsString());
  EXPECT_EQ("5x?x20", shape({5, kDynamic, 20}).getAsString());
  EXPECT_EQ("5x10x?", shape({5, 10, kDynamic}).getAsString());
  EXPECT_EQ("?", shape({kDynamic}).getAsString());
}

TEST(TensorShape, parsing) {
  EXPECT_EQ(ok(shape({})), TensorShape::parseFromString(""));
  EXPECT_EQ(ok(shape({5})), TensorShape::parseFromString("5"));
  EXPECT_EQ(ok(shape({5, 10})), TensorShape::parseFromString("5x10"));
  EXPECT_EQ(ok(shape({5, 10, 20})), TensorShape::parseFromString("5x10x20"));
  EXPECT_EQ(ok(shape({kDynamic, 10, 20})),
            TensorShape::parseFromString("?x10x20"));
  EXPECT_EQ(ok(shape({5, kDynamic, 20})),
            TensorShape::parseFromString("5x?x20"));
  EXPECT_EQ(ok(shape({5, 10, kDynamic})),
            TensorShape::parseFromString("5x10x?"));
  EXPECT_EQ(ok(shape({kDynamic})), TensorShape::parseFromString("?"));
  EXPECT_EQ(ErrorOr<TensorShape>(
                Error("could not parse dimension integer from string: 2x3.5 "
                      "because 3.5 cannot be parsed as an integer")),
            TensorShape::parseFromString("2x3.5"));
}

TEST(TensorShape, yamlInput) {
  llvm::yaml::Input input("shape: 5x?");
  ShapeHolder holder;
  input >> holder;
  EXPECT_EQ(shape({5, kDynamic}), holder.shape);
}

TEST(TensorShape, yamlOutput) {
  std::string str;
  llvm::raw_string_ostream os(str);
  llvm::yaml::Output output(os);
  ShapeHolder holder{shape({5, kDynamic})};
  output << holder;
  EXPECT_EQ(str, "---\nshape:           5x?\n...\n");
}
