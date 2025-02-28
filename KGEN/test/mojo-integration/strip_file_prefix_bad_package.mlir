// RUN: not mojo package %S/inputs/bad_package -kgenModule -strip-file-prefix=. 2>&1 | FileCheck %s
// CHECK: {{^}}inputs/bad_package/bad_file.mojo:7:5: error
