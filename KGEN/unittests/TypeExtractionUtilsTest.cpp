//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoTooling/TypeExtractionUtils.h"
#include "KGEN/MojoTooling/PublicASTDecl.h"
#include "KGEN/MojoTooling/TypeMetadata.h"
#include "gtest/gtest.h"

using namespace M;
using namespace M::KGEN::TypeExtractionUtils;

/// Test fixture for TypeExtractionUtils and TypeMetadata functionality.
class TypeExtractionUtilsTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

//===----------------------------------------------------------------------===//
// extractBaseTypeName function tests
//===----------------------------------------------------------------------===//

/// Test extractBaseTypeName with simple, non-generic types.
TEST_F(TypeExtractionUtilsTest, ExtractBaseTypeName_SimpleTypes) {
  EXPECT_EQ(extractBaseTypeName("Int"), "Int");
  EXPECT_EQ(extractBaseTypeName("String"), "String");
  EXPECT_EQ(extractBaseTypeName("Bool"), "Bool");
}

/// Test extractBaseTypeName with generic types containing type parameters.
/// Should strip generic parameters and return only the base type name.
TEST_F(TypeExtractionUtilsTest, ExtractBaseTypeName_GenericTypes) {
  EXPECT_EQ(extractBaseTypeName("List[Int]"), "List");
  EXPECT_EQ(extractBaseTypeName("Dict[String, Int]"), "Dict");
  EXPECT_EQ(extractBaseTypeName("Optional[Float64]"), "Optional");
  EXPECT_EQ(extractBaseTypeName("SIMD[DType.float32, 8]"), "SIMD");
}

/// Test extractBaseTypeName with nested generic types.
/// Should handle deeply nested generics and return only the outermost type.
TEST_F(TypeExtractionUtilsTest, ExtractBaseTypeName_NestedGenerics) {
  EXPECT_EQ(extractBaseTypeName("List[Dict[String, Int]]"), "List");
  EXPECT_EQ(extractBaseTypeName("Optional[List[String]]"), "Optional");
}

/// Test extractBaseTypeName with function types.
/// Function types should be preserved as-is without modification.
TEST_F(TypeExtractionUtilsTest, ExtractBaseTypeName_FunctionTypes) {
  EXPECT_EQ(extractBaseTypeName("fn(Int) -> String"), "fn(Int) -> String");
  EXPECT_EQ(extractBaseTypeName("fn(List[Int]) -> Bool"),
            "fn(List[Int]) -> Bool");
  EXPECT_EQ(extractBaseTypeName("fn() -> None"), "fn() -> None");
}

/// Test extractBaseTypeName with qualified type names.
/// Should preserve the full qualified name when generics are present.
TEST_F(TypeExtractionUtilsTest, ExtractBaseTypeName_QualifiedTypes) {
  EXPECT_EQ(extractBaseTypeName("std.collections.List"),
            "std.collections.List");
  EXPECT_EQ(extractBaseTypeName("std.collections.List[Int]"),
            "std.collections.List");
}

/// Test extractBaseTypeName with types containing whitespace.
/// Should properly trim whitespace from type names.
TEST_F(TypeExtractionUtilsTest, ExtractBaseTypeName_WithWhitespace) {
  EXPECT_EQ(extractBaseTypeName("  List[Int]  "), "List");
  EXPECT_EQ(extractBaseTypeName("\tDict[String, Int]\n"), "Dict");
}

/// Test extractBaseTypeName with OriginSet type.
/// Should return "OriginSet" instead of internal representation.
TEST_F(TypeExtractionUtilsTest, ExtractBaseTypeName_OriginSetType) {
  // Test OriginSet type name formatting (string-based fallback)
  EXPECT_EQ(extractBaseTypeName("origin.set"), "origin.set");
  EXPECT_EQ(extractBaseTypeName("OriginSet"), "OriginSet");
  EXPECT_EQ(extractBaseTypeName("OriginSet[something]"), "OriginSet");
}

/// Test extractBaseTypeName with edge cases and malformed inputs.
TEST_F(TypeExtractionUtilsTest, ExtractBaseTypeName_EdgeCases) {
  EXPECT_EQ(extractBaseTypeName(""), "");
  EXPECT_EQ(extractBaseTypeName("T"), "T");
  EXPECT_EQ(extractBaseTypeName("SomeType[]"), "SomeType");
}

//===----------------------------------------------------------------------===//
// generateDocPath function tests
//===----------------------------------------------------------------------===//

/// Test generateDocPath with standard library types.
/// Should generate proper std documentation paths.
TEST_F(TypeExtractionUtilsTest, GenerateDocPath_StandardlibTypes) {
  EXPECT_EQ(generateDocPath("std.collections", "List", ""),
            "/std/collections/List");
  EXPECT_EQ(generateDocPath("std.builtin", "Int", ""), "/std/builtin/Int");
}

/// Test generateDocPath with custom documentation base paths.
/// Should properly combine base paths with module and type names.
TEST_F(TypeExtractionUtilsTest, GenerateDocPath_WithDocsBasePath) {
  EXPECT_EQ(generateDocPath("collections", "List", "kernels"),
            "/kernels/collections/List");
  EXPECT_EQ(generateDocPath("", "CustomType", "mylib"), "/mylib/CustomType");
}

/// Test generateDocPath with special index module handling.
/// Should handle .index suffix for website compatibility (std only).
TEST_F(TypeExtractionUtilsTest, GenerateDocPath_IndexModules) {
  // Test .index suffix handling for website compatibility (std only)
  EXPECT_EQ(generateDocPath("std.utils.index", "IndexList", ""),
            "/std/utils/index_/IndexList");
  // Non-std modules should not get the .index_ conversion
  EXPECT_EQ(generateDocPath("mypackage.index", "SomeType", "docs"),
            "/docs/mypackage/index/SomeType");
}

/// Test generateDocPath with alias types.
/// Should generate paths with a lowercase anchor tag.
TEST_F(TypeExtractionUtilsTest, GenerateDocPath_Aliases) {
  EXPECT_EQ(generateDocPath("std.builtin", "MutOrigin", "", true),
            "/std/builtin/#mutorigin");
  EXPECT_EQ(generateDocPath("", "MyAlias", "", true), "//#myalias");
  EXPECT_EQ(generateDocPath("mymodule", "SomeAlias", "docs", true),
            "/docs/mymodule/#somealias");
}

/// Test generateDocPath with edge cases and empty inputs.
TEST_F(TypeExtractionUtilsTest, GenerateDocPath_EdgeCases) {
  EXPECT_EQ(generateDocPath("", "", ""), "");
  EXPECT_EQ(generateDocPath("module", "", ""), "");
  EXPECT_EQ(generateDocPath("", "Type", ""), "//Type");
}

/// Test generateDocPath with __init__ module removal.
/// Should remove __init__ components from module paths for cleaner
/// documentation URLs.
TEST_F(TypeExtractionUtilsTest, GenerateDocPath_InitModuleRemoval) {
  // __init__ at the end of a path should be removed
  EXPECT_EQ(generateDocPath("std.collections.__init__", "List", ""),
            "/std/collections/List");

  // __init__ in the middle of a path should be removed
  EXPECT_EQ(generateDocPath("std.__init__.collections", "Dict", ""),
            "/std/collections/Dict");

  // __init__ with aliases should work correctly
  EXPECT_EQ(generateDocPath("std.builtin.__init__", "MutOrigin", "", true),
            "/std/builtin/#mutorigin");
}

/// Test generateDocPath dot-to-slash conversion for nested modules.
/// Should properly convert module.submodule notation to path separators.
TEST_F(TypeExtractionUtilsTest, GenerateDocPath_DotToSlashConversion) {
  EXPECT_EQ(generateDocPath("a.b.c.d", "Type", ""), "/a/b/c/d/Type");
  EXPECT_EQ(generateDocPath("deeply.nested.module.hierarchy", "Type", "base"),
            "/base/deeply/nested/module/hierarchy/Type");
}

//===----------------------------------------------------------------------===//
// extractLibraryInfo function tests
//===----------------------------------------------------------------------===//

/// Test extractLibraryInfo with simple types (no AST context).
/// Should return basic metadata without documentation paths.
TEST_F(TypeExtractionUtilsTest, ExtractLibraryInfo_SimpleTypes) {
  // Test basic type extraction without AST context
  auto metadata = extractLibraryInfo("Int");
  auto json = metadata.toJSON();

  EXPECT_TRUE(json.find("type") != json.end());
  auto typeValue = json.find("type")->second.getAsString();
  EXPECT_TRUE(typeValue.has_value());
  EXPECT_EQ(typeValue.value(), "Int");

  // Should not have path for unknown types
  EXPECT_TRUE(json.find("path") == json.end());
}

/// Test extractLibraryInfo with generic types.
/// Should strip generic parameters in the returned metadata.
TEST_F(TypeExtractionUtilsTest, ExtractLibraryInfo_GenericTypes) {
  // Test that generic parameters are stripped
  auto metadata = extractLibraryInfo("List[Int]");
  auto json = metadata.toJSON();

  auto typeValue = json.find("type")->second.getAsString();
  EXPECT_TRUE(typeValue.has_value());
  EXPECT_EQ(typeValue.value(), "List");
}

/// Test extractLibraryInfo with qualified type names.
/// Should extract the leaf type name from qualified names.
TEST_F(TypeExtractionUtilsTest, ExtractLibraryInfo_QualifiedTypes) {
  // Test that qualified names get the leaf type
  auto metadata = extractLibraryInfo("Dict[String, Int]");
  auto json = metadata.toJSON();

  auto typeValue = json.find("type")->second.getAsString();
  EXPECT_TRUE(typeValue.has_value());
  EXPECT_EQ(typeValue.value(), "Dict");
}

/// Test extractLibraryInfo with function types.
/// Should preserve function signatures without modification.
TEST_F(TypeExtractionUtilsTest, ExtractLibraryInfo_FunctionTypes) {
  // Test that function types are preserved as-is
  auto metadata = extractLibraryInfo("fn(Int) -> String");
  auto json = metadata.toJSON();

  auto typeValue = json.find("type")->second.getAsString();
  EXPECT_TRUE(typeValue.has_value());
  EXPECT_EQ(typeValue.value(), "fn(Int) -> String");
}

/// Test extractLibraryInfo with empty input.
TEST_F(TypeExtractionUtilsTest, ExtractLibraryInfo_EmptyType) {
  // Test edge case with empty type
  auto metadata = extractLibraryInfo("");
  auto json = metadata.toJSON();

  auto typeValue = json.find("type")->second.getAsString();
  EXPECT_TRUE(typeValue.has_value());
  EXPECT_EQ(typeValue.value(), "");
}

//===----------------------------------------------------------------------===//
// TypeMetadata class tests
//===----------------------------------------------------------------------===//

/// Test TypeMetadata JSON serialization with complete metadata.
/// Should include both type and path information in JSON output.
TEST_F(TypeExtractionUtilsTest, TypeMetadata_ToJSON) {
  // Test TypeMetadata JSON serialization
  // Constructor: TypeMetadata(typeStr, module, relativePath, constraints)
  M::KGEN::TypeMetadata metadata("List[Int]", "std.collections",
                                 "/std/collections/List");

  auto json = metadata.toJSON();

  EXPECT_TRUE(json.find("type") != json.end());
  EXPECT_TRUE(json.find("path") != json.end());

  auto typeValue = json.find("type")->second.getAsString();
  auto pathValue = json.find("path")->second.getAsString();

  EXPECT_TRUE(typeValue.has_value());
  EXPECT_TRUE(pathValue.has_value());
  EXPECT_EQ(typeValue.value(), "List[Int]");
  EXPECT_EQ(pathValue.value(), "/std/collections/List");
}

/// Test TypeMetadata JSON serialization with empty path.
/// Should omit path field when empty (not include empty string).
TEST_F(TypeExtractionUtilsTest, TypeMetadata_EmptyPath) {
  // Test with empty path (should not include path in JSON)
  // Constructor: TypeMetadata(typeStr, module, relativePath, constraints)
  M::KGEN::TypeMetadata metadata("SomeType", "", "");

  auto json = metadata.toJSON();

  EXPECT_TRUE(json.find("type") != json.end());
  EXPECT_TRUE(json.find("path") == json.end());
}
