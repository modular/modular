//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace M;
using namespace mlir;

TEST(MAttrs, createResourceAttr) {
  MLIRContext ctx;
  DenseResourceElementsAttr attr;
  {
    // The underlying string is released, but the resource attribute copies it.
    std::string data = "Please pretend this is MLIR bytecode.";
    attr = createResourceAttr(&ctx, ArrayRef(data.c_str(), data.size()),
                              "This is the name.");
  }
  EXPECT_EQ(attr.getRawHandle().getKey(), "This is the name.");
  EXPECT_EQ(std::string(attr.getRawHandle().getBlob()->getData().data()),
            "Please pretend this is MLIR bytecode.");
}
