//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Package/Package.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace M;
using namespace KGEN;
using namespace mlir;

TEST(Package, createElaboratedBytecodeAttr) {
  MLIRContext ctx;
  ctx.loadDialect<KGENDialect>();

  Location loc = UnknownLoc::get(&ctx);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder b(module->getBody(), module->getBody()->begin());

  auto func = b.create<FuncOp>(loc, b.getStringAttr("foo"),
                               SignatureType::get(&ctx, {}, {}));

  // Technically this function expects an elaborated module, but it doesn't
  // check.
  SymbolTable symtab(*module);
  auto bytecodeOr = createElaboratedBytecodeAttr(
      symtab, FlatSymbolRefAttr::get(StringAttr::get(&ctx, "bar")));

  // The module bytecode is stored in the returned attribute.
  ASSERT_FALSE(bytecodeOr.isError());
  EXPECT_THAT(bytecodeOr->getRawHandle().getKey().str(),
              testing::StartsWith("bytecode_"));
  // Functions in the module are exported.
  EXPECT_TRUE(func.isExported());
}
