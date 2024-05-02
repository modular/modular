//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(SignatureHelpTest, testSignatureHelpOverload) {
  Document doc("test:///foo.mojo", R"(
fn function(): # skip
    return
fn function(arg: Int) -> Int: # skip
    return arg
fn function(arg: Bool, arg2: Int) -> Int: # skip
    return arg2

fn test():
    function()
    function(10)
    function(True, 10)
)");

  std::vector<lsp::Range> ranges = doc.findAllRanges("function(");
  ASSERT_EQ((int)ranges.size(), 3);

  createTestClient()
      .open(doc)
      .signatureHelp(doc, ranges[0].end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 3);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 0);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "fn function()");
                       EXPECT_EQ(signatureHelp.signatures[1].label,
                                 "fn function(arg: Int) -> Int");
                       EXPECT_EQ(signatureHelp.signatures[2].label,
                                 "fn function(arg: Bool, arg2: Int) -> Int");
                     })
      .signatureHelp(doc, ranges[1].end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 2);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 0);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "fn function(arg: Int) -> Int");
                       EXPECT_EQ(signatureHelp.signatures[1].label,
                                 "fn function(arg: Bool, arg2: Int) -> Int");
                     })
      .signatureHelp(doc, ranges[2].end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 1);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 0);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "fn function(arg: Bool, arg2: Int) -> Int");
                     })
      .signatureHelp(doc, doc.findLastRange("True,")->end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 1);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 1);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "fn function(arg: Bool, arg2: Int) -> Int");
                     })
      .execute();
}

TEST(SignatureHelpTest, testSignatureHelpTypeCall) {
  Document doc("test:///foo.mojo", R"(
struct SomeStruct:
    var a_field: Int

    fn __init__(inout self):
        pass

    fn __init__(inout self, a_field: Int):
        pass

fn test():
    SomeStruct()
)");

  createTestClient()
      .open(doc)
      .signatureHelp(doc, doc.findLastRange("SomeStruct(")->end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 2);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 1);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "fn __init__(inout self: Self, /)");
                       EXPECT_EQ(
                           signatureHelp.signatures[1].label,
                           "fn __init__(inout self: Self, /, a_field: Int)");
                     })
      .execute();
}

TEST(SignatureHelpTest, testSignatureOverloadParams) {
  Document doc("test:///foo.mojo", R"(
fn function[type: DType](): # skip
    return
fn function[type: DType, type2: DType](): # skip
    return

fn test():
    function[DType.bool]()
    function[DType.bool, DType.bool]()
)");

  createTestClient()
      .open(doc)
      .signatureHelp(doc, doc.findFirstRange("function[")->end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 2);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 0);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "fn function[type: DType]()");
                       EXPECT_EQ(signatureHelp.signatures[1].label,
                                 "fn function[type: DType, type2: DType]()");
                     })
      .signatureHelp(doc, doc.findLastRange("DType.bool,")->end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 1);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 1);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "fn function[type: DType, type2: DType]()");
                     })
      .execute();
}

TEST(SignatureHelpTest, testSignatureHelpParams) {
  Document doc("test:///foo.mojo", R"(
struct SomeStruct[dtype: DType]: # skip
    fn __init__(inout self):
        pass

fn test():
    SomeStruct[DType.bool]()
)");

  createTestClient()
      .open(doc)
      .signatureHelp(doc, doc.findLastRange("SomeStruct[")->end,
                     [](const lsp::SignatureHelp2 &signatureHelp) {
                       ASSERT_EQ((int)signatureHelp.signatures.size(), 1);
                       EXPECT_EQ(signatureHelp.activeSignature, 0);
                       EXPECT_EQ(signatureHelp.activeParameter, 0);
                       EXPECT_EQ(signatureHelp.signatures[0].label,
                                 "struct SomeStruct[dtype: DType]");
                     })
      .execute();
}
