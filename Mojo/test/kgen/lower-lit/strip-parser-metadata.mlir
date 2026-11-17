// RUN: kgen-opt -strip-parser-metadata %s | FileCheck %s

// CHECK-NOT: #doc_string
#doc_string = #lit.doc.string<"Package docstring">
#doc_string1 = #lit.doc.string<"Module docstring">
module {
  lit.package @package {
    lit.file_module @module {
    } <docString = #doc_string1>
  } <docString = #doc_string>
}
