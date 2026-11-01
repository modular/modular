"""The symbols libmax exports.

Several GraphCompiler libraries build with -fvisibility=default so the Python
extensions can bind to their dialects without statically linking them. The
version script (Linux) and exported symbols list (macOS) built from this list
narrow the export table back to what consumers call.

Entries are mangled ELF names or globs; the macOS list adds Mach-O's extra
leading underscore. A name without glob characters must be defined, or the
link fails.
"""

LIBMAX_EXPORTS = [
    # The MAX and MLIR C APIs.
    "M_*",
    "mlir*",

    # Called from compiled Mojo code through `external_call`, which resolves
    # them by name.
    "concat_mojo_strings",
    "create_mojo_string",
    "destroy_mojo_string",
    "get_mojo_string_data",
    "get_mojo_string_length",
    "create_trace",
    "destroy_trace",
    "end_trace",
    "mark_trace",
    "start_trace",
    "create_weights_registry",
    "destroy_weights_registry",
    "weights_registry_lookup",
    "weights_registry_lookup_mojo_string",

    # The GDB JIT interface, which debuggers find by name.
    "__jit_debug_descriptor",
    "__jit_debug_register_code",

    # opentelemetry's runtime context. Every DSO that includes its headers
    # defines these, and exporting them is what makes the copies one.
    "_ZZN13opentelemetry2v17context14RuntimeContext10GetStorageEvE7context",
    "_ZGVZN13opentelemetry2v17context14RuntimeContext10GetStorageEvE7context",
    "_ZZN13opentelemetry2v17context25ThreadLocalContextStorage8GetStackEvE6stack_",
    "_ZGVZN13opentelemetry2v17context25ThreadLocalContextStorage8GetStackEvE6stack_",

    # The device driver. _core links its own copy, so nothing imports this, but
    # exporting it from both is what makes the two copies one: one device
    # context cache, one device runtime. Hidden here, a process ends up with
    # two, and tearing both down at exit crashes under the profiler.
    "_ZN1M6Driver*",
    "_ZNK1M6Driver*",

    # The C++ engine API used by the Python bindings and the mach shim. `NK` is
    # the const-member form of `N`.
    "_ZN1M14CompiledModels*",
    "_ZNK1M14CompiledModels*",
    "_ZN1M6Engine*",
    "_ZNK1M6Engine*",
    "_ZN1M15ExecutableModel*",
    "_ZNK1M15ExecutableModel*",
    "_ZN1M13GraphCompiler*",
    "_ZNK1M13GraphCompiler*",
    "_ZN1M10MOFContext*",
    "_ZNK1M10MOFContext*",
    "_ZN1M10SafeTensor*",
    "_ZNK1M10SafeTensor*",
    "_ZN1M9Telemetry*",
    "_ZNK1M9Telemetry*",
    # Functions directly in `M`; classes are capitalized.
    "_ZN1M[1-9][a-z]*",
    "_ZN1M[1-9][0-9][a-z]*",
    "_Z[1-9]MAXG_*",
    "_Z[1-9][0-9]MAXG_*",

    # Hand-written dialect API used by the Python bindings, including every pass
    # constructor.
    "_ZN1M2MO10toMLIRTypeE*",
    "_ZN1M2MO12fromMLIRTypeE*",
    "_ZN1M2MO12loadDialectsE*",
    "_ZN1M2MO[1-9][0-9]create*",
    "_ZN1M3RMO[1-9][0-9]create*",
    "_ZN1M3RMO31materializePythonFrameLocationsE*",
    "_ZN1M4MOGG14verifyCustomOpE*",
    "_ZN1M4MOGG12MojoRegistry*",
    "_ZNK1M4MOGG12MojoRegistry*",
    "_ZN1M4MOGG17KernelDeclAdaptor*",
    "_ZNK1M4MOGG17KernelDeclAdaptor*",

    # The ODS API the generated bindings in
    # max/python/max/_core/internal/modules/dialects call, for every op,
    # attribute and type class: its name is capitalized, which leaves out
    # `detail` and `impl`. The class name's length prefix takes one or two
    # digits.
    #
    # Only builders taking an `OpBuilder&`; the `ImplicitLocOpBuilder&` half
    # goes unused. Of the getters, the parameterless accessors plus the few the
    # builders call. `inferReturnTypes` and `setPropertiesFromAttr` serve only
    # the builders of ops that infer their result types, which only RMO binds.
    "_ZN1M2MO[1-9][A-Z]*6createERN4mlir9OpBuilderE*",
    "_ZN1M2MO[1-9][A-Z]*27getODSOperandIndexAndLengthEj",
    "_ZN1M2MO[1-9][A-Z]*3getE*",
    "_ZN1M2MO[1-9][A-Z]*10getCheckedE*",
    "_ZN1M2MO[1-9][A-Z]*[0-9]get*Ev",
    "_ZNK1M2MO[1-9][A-Z]*[0-9]get*Ev",
    "_ZN1M2MO[1-9][0-9][A-Z]*6createERN4mlir9OpBuilderE*",
    "_ZN1M2MO[1-9][0-9][A-Z]*27getODSOperandIndexAndLengthEj",
    "_ZN1M2MO[1-9][0-9][A-Z]*3getE*",
    "_ZN1M2MO[1-9][0-9][A-Z]*10getCheckedE*",
    "_ZN1M2MO[1-9][0-9][A-Z]*[0-9]get*Ev",
    "_ZNK1M2MO[1-9][0-9][A-Z]*[0-9]get*Ev",
    "_ZN1M3RMO[1-9][A-Z]*6createERN4mlir9OpBuilderE*",
    "_ZN1M3RMO[1-9][A-Z]*16inferReturnTypesE*",
    "_ZN1M3RMO[1-9][A-Z]*21setPropertiesFromAttrE*",
    "_ZN1M3RMO[1-9][A-Z]*27getODSOperandIndexAndLengthEj",
    "_ZN1M3RMO[1-9][A-Z]*3getE*",
    "_ZN1M3RMO[1-9][A-Z]*10getCheckedE*",
    "_ZN1M3RMO[1-9][A-Z]*[0-9]get*Ev",
    "_ZNK1M3RMO[1-9][A-Z]*[0-9]get*Ev",
    "_ZN1M3RMO[1-9][0-9][A-Z]*6createERN4mlir9OpBuilderE*",
    "_ZN1M3RMO[1-9][0-9][A-Z]*16inferReturnTypesE*",
    "_ZN1M3RMO[1-9][0-9][A-Z]*21setPropertiesFromAttrE*",
    "_ZN1M3RMO[1-9][0-9][A-Z]*27getODSOperandIndexAndLengthEj",
    "_ZN1M3RMO[1-9][0-9][A-Z]*3getE*",
    "_ZN1M3RMO[1-9][0-9][A-Z]*10getCheckedE*",
    "_ZN1M3RMO[1-9][0-9][A-Z]*[0-9]get*Ev",
    "_ZNK1M3RMO[1-9][0-9][A-Z]*[0-9]get*Ev",
    "_ZN1M4MOSH[1-9][A-Z]*6createERN4mlir9OpBuilderE*",
    "_ZN1M4MOSH[1-9][A-Z]*27getODSOperandIndexAndLengthEj",
    "_ZN1M4MOSH[1-9][A-Z]*3getE*",
    "_ZN1M4MOSH[1-9][A-Z]*10getCheckedE*",
    "_ZN1M4MOSH[1-9][A-Z]*[0-9]get*Ev",
    "_ZNK1M4MOSH[1-9][A-Z]*[0-9]get*Ev",
    "_ZN1M4MOSH[1-9][0-9][A-Z]*6createERN4mlir9OpBuilderE*",
    "_ZN1M4MOSH[1-9][0-9][A-Z]*27getODSOperandIndexAndLengthEj",
    "_ZN1M4MOSH[1-9][0-9][A-Z]*3getE*",
    "_ZN1M4MOSH[1-9][0-9][A-Z]*10getCheckedE*",
    "_ZN1M4MOSH[1-9][0-9][A-Z]*[0-9]get*Ev",
    "_ZNK1M4MOSH[1-9][0-9][A-Z]*[0-9]get*Ev",
    # MOGG has no bindings module; the hand-written bindings use two accessors.
    "_ZN1M4MOGG[1-9][0-9][A-Z]*15getRegistrationEv",
    "_ZN1M4MOGG24ProfileScopeLocationAttr3getE*",
]
