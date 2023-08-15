# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Invoking the subcommand with `--help-text` prints its help text.
# RUN: mojo demangle --one -two --help-text | FileCheck %s --check-prefix CHECK-HELP
# CHECK-HELP: Demangles the given name

# Reject unknown options.
# RUN: not mojo demangle -one --two '$aModule::main()' 2>&1 | FileCheck %s --check-prefix CHECK-UNKNOWN
# CHECK-UNKNOWN: mojo{{.*}}: error: unrecognized argument '-one'

# RUN: mojo demangle '$aModule::main()' | FileCheck -check-prefix="SIMPLE" %s
# SIMPLE: Mangled: "$aModule::main()" - Modules: ["aModule"], Structs: [], Symbol: "main", Identifier: "main", Signature: () -> ()

# Names can be passed in via stdin.
# RUN: echo "\$aModule::main()" | mojo demangle | FileCheck %s --check-prefix SIMPLE

# RUN: mojo demangle '$aModule::AStruct::main()' | FileCheck -check-prefix="STRUCT" %s
# STRUCT: Mangled: "$aModule::AStruct::main()" - Modules: ["aModule"], Structs: ["AStruct"], Symbol: "main", Identifier: "main", Signature: () -> ()

# RUN: mojo demangle '$aPackage::$aModule::AStruct::BStruct::main()' | FileCheck -check-prefix="NESTED" %s
# NESTED: Mangled: "$aPackage::$aModule::AStruct::BStruct::main()" - Modules: ["aPackage", "aModule"], Structs: ["AStruct", "BStruct"], Symbol: "main", Identifier: "main", Signature: () -> ()

# RUN: mojo demangle '$aModule::AStruct::BStruct::main(index,!pop.struct<dtype>)' | FileCheck -check-prefix="NESTED2" %s
# NESTED2: Mangled: "$aModule::AStruct::BStruct::main(index,!pop.struct<dtype>)" - Modules: ["aModule"], Structs: ["AStruct", "BStruct"], Symbol: "main", Identifier: "main", Signature: (index, !pop.struct<dtype>) -> ()

# RUN: mojo demangle 'AStruct::main()' | FileCheck -check-prefix="NOMODULE" %s
# NOMODULE: Mangled: "AStruct::main()" - Modules: [], Structs: ["AStruct"], Symbol: "main", Identifier: "main", Signature: () -> ()

# RUN: mojo demangle 'main()' | FileCheck -check-prefix="NOMODULE2" %s
# NOMODULE2: Mangled: "main()" - Modules: [], Structs: [], Symbol: "main", Identifier: "main", Signature: () -> ()

# RUN: mojo demangle 'main(index,index)!pop.struct<index,index>' | FileCheck -check-prefix="NOMODULE3" %s
# NOMODULE3: Mangled: "main(index,index)!pop.struct<index,index>" - Modules: [], Structs: [], Symbol: "main", Identifier: "main", Signature: (index, index) -> !pop.struct<index, index>

# RUN: mojo demangle 'main()!pop.struct<index,index>' | FileCheck -check-prefix="NOMODULE4" %s
# NOMODULE4: Mangled: "main()!pop.struct<index,index>" - Modules: [], Structs: [], Symbol: "main", Identifier: "main", Signature: () -> !pop.struct<index, index>

# RUN: mojo demangle '$aModule::AStruct' | FileCheck -check-prefix="NOFUNC" %s
# NOFUNC: Mangled: "$aModule::AStruct" - Modules: ["aModule"], Structs: [], Symbol: "AStruct", Identifier: "AStruct", Signature: (none)

# RUN: mojo demangle '$aModule::AStruct::BStruct' | FileCheck -check-prefix="NESTEDNOFUNC" %s
# NESTEDNOFUNC: Mangled: "$aModule::AStruct::BStruct" - Modules: ["aModule"], Structs: ["AStruct"], Symbol: "BStruct", Identifier: "BStruct", Signature: (none)

# RUN: mojo demangle 'Mod::AStruct::main($Int::Int)' | FileCheck -check-prefix=MANGLEDTYPE  %s
# MANGLEDTYPE: Mangled: "Mod::AStruct::main($Int::Int)" - Modules: [], Structs: ["Mod", "AStruct"], Symbol: "main", Identifier: "main"

# RUN: mojo demangle 'main($functions::SomeStruct[size, other_param]&)' | FileCheck -check-prefix=PARAMETRIZEDARG %s
# PARAMETRIZEDARG: Mangled: "main($functions::SomeStruct[size, other_param]&)" - Modules: [], Structs: [], Symbol: "main", Identifier: "main", Signature: (none)

# RUN: mojo demangle '$AModule::AStruct::print[$builtin::$Int::Int,$DType::DType]($builtin::$SIMD::SIMD[*(0,1), *(0,0)])' | FileCheck -check-prefix=PARAMETRIZEDARG2 %s
# PARAMETRIZEDARG2: Mangled: "$AModule::AStruct::print[$builtin::$Int::Int,$DType::DType]($builtin::$SIMD::SIMD[*(0,1), *(0,0)])" - Modules: ["AModule"], Structs: ["AStruct"], Symbol: "print[$builtin::$Int::Int,$DType::DType]", Identifier: "print", Signature: (none)

# Demangling failures are printed to stderr.
# RUN: not mojo demangle '$aModule::AStruct::BStruct(!invalid.type)' 2>&1 | FileCheck -check-prefix="FAILURE" %s
# FAILURE: demangling failed

# Only one name at a time.
# RUN: not mojo demangle 'one' 'two' 2>&1 | FileCheck --check-prefix TOO-MANY %s
# TOO-MANY: cannot demangle both 'one' and 'two'

# An empty string can be demangled.
# RUN: mojo demangle "" | FileCheck %s --check-prefix EMPTY
# EMPTY: Mangled: "" - Modules: [], Structs: [], Symbol: "", Identifier: "", Signature: (none)
