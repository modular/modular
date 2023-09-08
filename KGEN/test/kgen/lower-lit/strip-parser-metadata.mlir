// RUN: kgen-opt -strip-parser-metadata %s | FileCheck %s

// CHECK-NOT: #doc_string
#doc_string = #lit.doc.string<"Package docstring">
#doc_string1 = #lit.doc.string<"Module docstrig">
module {
  lit.package @package attributes {archiveBytes = dense_resource<archive_0> : tensor<4086xui8>, compiledFor = #M.target<triple = "", cpu = "", features = "", data_layout = "", simd_bit_width = 256>, docString = #doc_string, postElaborationModule = dense_resource<bytecode_0> : tensor<2893xui8>, preElaborationModule = dense_resource<bytecode_1> : tensor<7468xui8>} {
    lit.file_module @module attributes {docString = #doc_string1} {
    }
  }
}

// CHECK-NOT: dialect_resources
{-#
  dialect_resources: {
    builtin: {
      archive_0: "0x08000000",
      bytecode_0: "0x08000000",
      bytecode_1: "0x08000000"
    }
  }
#-}
