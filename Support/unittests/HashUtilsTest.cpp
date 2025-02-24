//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HashUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace M;

TEST(HashUtils, GetBytecodeHashBasic) {
  MLIRContext context;
  OpBuilder builder(&context);

  // Create a simple operation
  auto loc = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp = mlir::ModuleOp::create(loc);

  FailureOr<std::string> result = getBytecodeHash(*moduleOp);
  ASSERT_TRUE(succeeded(result));
  EXPECT_FALSE(result->empty());

  // Hash should be 32 chars (128 bits = 16 bytes = 32 hex chars)
  // NB: As of 2025-02-22, the hash is actually 31 chars, but the implementation
  // zero pads to 32.
  ASSERT_EQ(result->size(), 32);
}

TEST(HashUtils, GetBytecodeHashEquivalentOps) {
  MLIRContext context;
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  // Create two equivalent operations
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp1 = mlir::ModuleOp::create(loc);
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp2 = mlir::ModuleOp::create(loc);

  FailureOr<std::string> hash1 = getBytecodeHash(*moduleOp1);
  FailureOr<std::string> hash2 = getBytecodeHash(*moduleOp2);

  EXPECT_TRUE(succeeded(hash1));
  EXPECT_TRUE(succeeded(hash2));

  // Equivalent ops should have same hash
  ASSERT_EQ(*hash1, *hash2);
}

TEST(HashUtils, GetBytecodeHashDifferentOps) {
  MLIRContext context;
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  // Create two different operations
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp1 = mlir::ModuleOp::create(loc);
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp2 = mlir::ModuleOp::create(loc);

  // Add something to moduleOp2 to make it different
  builder.setInsertionPointToStart(moduleOp2->getBody());
  builder.create<ModuleOp>(loc);

  FailureOr<std::string> hash1 = getBytecodeHash(*moduleOp1);
  FailureOr<std::string> hash2 = getBytecodeHash(*moduleOp2);

  EXPECT_TRUE(succeeded(hash1));
  EXPECT_TRUE(succeeded(hash2));

  // Different ops should have different hashes
  ASSERT_NE(*hash1, *hash2);
}

TEST(HashUtils, GetBytecodeHashIgnoresLocation) {
  MLIRContext context;
  OpBuilder builder(&context);

  // Create two ops with different locations
  auto loc1 = builder.getUnknownLoc();
  auto loc2 = FileLineColLoc::get(builder.getStringAttr("test.mlir"), 1, 1);

  mlir::OwningOpRef<mlir::ModuleOp> moduleOp1 = mlir::ModuleOp::create(loc1);
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp2 = mlir::ModuleOp::create(loc2);

  FailureOr<std::string> hash1 = getBytecodeHash(*moduleOp1);
  FailureOr<std::string> hash2 = getBytecodeHash(*moduleOp2);

  EXPECT_TRUE(succeeded(hash1));
  EXPECT_TRUE(succeeded(hash2));

  // Hashes should be equal despite different locations
  ASSERT_EQ(*hash1, *hash2);
}
