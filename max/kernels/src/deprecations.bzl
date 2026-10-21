"""Deprecated stdlib/kernel declarations that `max/kernels` and `Kernels`
intentionally still call: the raw-Pointer APIs.

Splat `IGNORED_POINTER_DEPRECATIONS` into a `mojo_library`'s
`ignore_deprecated` param, or `IGNORED_POINTER_DEPRECATIONS_COPTS` into a
`mojo_test`/`mojo_binary`/`mojo_filecheck_test`'s `copts`, to suppress just
these declarations' deprecation warnings for that target, while leaving all
other deprecation warnings visible.
"""

IGNORED_POINTER_DEPRECATIONS = [
    "Pointer.__getitem__",
]

IGNORED_POINTER_DEPRECATIONS_COPTS = [
    "--ignore-deprecated=" + decl_name
    for decl_name in IGNORED_POINTER_DEPRECATIONS
]
