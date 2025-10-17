"""Helper code for feature crosstool configuration."""

MARKER_FEATURES = [
    "archive_param_file",
    "compile_all_modules",
    "exclude_private_headers_in_module_maps",
    "gcc_quoting_for_param_files",
    "modular_code",  # Differentiate between third and first party code
    "module_maps",
    "only_doth_headers_in_module_maps",
    "supports_start_end_lib",
    "sysroot",  # Stop duplicate sysroot confusion until legacy features are disabled
]
