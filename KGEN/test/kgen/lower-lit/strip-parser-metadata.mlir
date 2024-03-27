// RUN: kgen-opt -strip-parser-metadata %s | FileCheck %s

// CHECK-NOT: #doc_string
#doc_string = #lit.doc.string<"Package docstring">
#doc_string1 = #lit.doc.string<"Module docstring">
module {
  lit.package @package attributes {docString = #doc_string, postParseModule = dense_resource<bytecode_1> : tensor<7468xui8>} {
    lit.file_module @module attributes {docString = #doc_string1} {
    }
  }
}

// CHECK-NOT: dialect_resources
{-#
  dialect_resources: {
    builtin: {
      bytecode_1: "0x08000000"
    }
  }
#-}
