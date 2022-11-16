// RUN: support-dialect-opt -allow-unregistered-dialect %s | support-dialect-opt -allow-unregistered-dialect | FileCheck %s

// COM: This ensures we can roundtrip a `cache.symbol`.
// CHECK-LABEL: cache.symbol @afunc
// CHECK-SAME: "func.func" regions=[
// CHECK-SAME:   #cache.region_hash<"YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo=">
// CHECK-SAME: ] original_attrs={function_type = () -> ()}
cache.symbol @afunc "func.func"
regions=[
  #cache.region_hash<"YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo=">
] original_attrs = {function_type=()->()}