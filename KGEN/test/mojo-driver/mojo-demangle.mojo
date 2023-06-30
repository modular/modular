# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Invoking the subcommand with `--help` prints its help text.
# RUN: mojo-driver demangle --one -two --help | FileCheck %s --check-prefix CHECK-HELP
# CHECK-HELP: Demangle the given name

# Reject unknown options.
# RUN: not mojo-driver demangle -one --two '$aModule::main()' 2>&1 | FileCheck %s --check-prefix CHECK-UNKNOWN
# CHECK-UNKNOWN: mojo-driver{{.*}}: error: unrecognized argument '-one'

# RUN: mojo-driver demangle '$aModule::main()' | FileCheck -check-prefix="SIMPLE" %s
# SIMPLE: Mangled: "$aModule::main()" - Modules: ["aModule"], Structs: [], Symbol: "main", Signature: () -> ()

# Names can be passed in via stdin.
# RUN: echo "\$aModule::main()" | mojo-driver demangle | FileCheck %s --check-prefix SIMPLE

# RUN: mojo-driver demangle '$aModule::AStruct::main()' | FileCheck -check-prefix="STRUCT" %s
# STRUCT: Mangled: "$aModule::AStruct::main()" - Modules: ["aModule"], Structs: ["AStruct"], Symbol: "main", Signature: () -> ()

# RUN: mojo-driver demangle '$aPackage::$aModule::AStruct::BStruct::main()' | FileCheck -check-prefix="NESTED" %s
# NESTED: Mangled: "$aPackage::$aModule::AStruct::BStruct::main()" - Modules: ["aPackage", "aModule"], Structs: ["AStruct", "BStruct"], Symbol: "main", Signature: () -> ()

# RUN: mojo-driver demangle '$aModule::AStruct::BStruct::main(index,!pop.struct<dtype>)' | FileCheck -check-prefix="NESTED2" %s
# NESTED2: Mangled: "$aModule::AStruct::BStruct::main(index,!pop.struct<dtype>)" - Modules: ["aModule"], Structs: ["AStruct", "BStruct"], Symbol: "main", Signature: (index, !pop.struct<dtype>) -> ()

# RUN: mojo-driver demangle 'AStruct::main()' | FileCheck -check-prefix="NOMODULE" %s
# NOMODULE: Mangled: "AStruct::main()" - Modules: [], Structs: ["AStruct"], Symbol: "main", Signature: () -> ()

# RUN: mojo-driver demangle 'main()' | FileCheck -check-prefix="NOMODULE2" %s
# NOMODULE2: Mangled: "main()" - Modules: [], Structs: [], Symbol: "main", Signature: () -> ()

# RUN: mojo-driver demangle 'main(index,index)!pop.struct<index,index>' | FileCheck -check-prefix="NOMODULE3" %s
# NOMODULE3: Mangled: "main(index,index)!pop.struct<index,index>" - Modules: [], Structs: [], Symbol: "main", Signature: (index, index) -> !pop.struct<index, index>

# RUN: mojo-driver demangle 'main()!pop.struct<index,index>' | FileCheck -check-prefix="NOMODULE4" %s
# NOMODULE4: Mangled: "main()!pop.struct<index,index>" - Modules: [], Structs: [], Symbol: "main", Signature: () -> !pop.struct<index, index>

# RUN: mojo-driver demangle '$aModule::AStruct' | FileCheck -check-prefix="NOFUNC" %s
# NOFUNC: Mangled: "$aModule::AStruct" - Modules: ["aModule"], Structs: [], Symbol: "AStruct", Signature: (none)

# RUN: mojo-driver demangle '$aModule::AStruct::BStruct' | FileCheck -check-prefix="NESTEDNOFUNC" %s
# NESTEDNOFUNC: Mangled: "$aModule::AStruct::BStruct" - Modules: ["aModule"], Structs: ["AStruct"], Symbol: "BStruct", Signature: (none)

# RUN: mojo-driver demangle 'Mod::AStruct::foo($Int::Int)' | FileCheck -check-prefix=MANGLEDTYPE  %s
# MANGLEDTYPE: Mangled: "Mod::AStruct::foo($Int::Int)" - Modules: [], Structs: ["Mod", "AStruct"], Symbol: "foo"

# Demangling failures are printed to stderr.
# RUN: not mojo-driver demangle '$aModule::AStruct::BStruct(!invalid.type)' 2>&1 | FileCheck -check-prefix="FAILURE" %s
# FAILURE: demangling failed

# Only one name at a time.
# RUN: not mojo-driver demangle 'one' 'two' 2>&1 | FileCheck --check-prefix TOO-MANY %s
# TOO-MANY: cannot demangle both 'one' and 'two'

# An empty string can be demangled.
# RUN: mojo-driver demangle "" | FileCheck %s --check-prefix EMPTY
# EMPTY: Mangled: "" - Modules: [], Structs: [], Symbol: "", Signature: (none)
