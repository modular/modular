"""Shared config that needs to be referenced by BUILD.bazel files"""

# GPU name, brandh, target-accelerator argument
SUPPORTED_GPUS = [
    ("a10", "nvidia", "86"),
    ("a100", "nvidia", "80"),
    ("a3000", "nvidia", "86"),
    ("l4", "nvidia", "89"),
    ("h100", "nvidia", "90a"),
    ("h200", "nvidia", "90a"),
    ("b100", "nvidia", "100a"),
    ("b200", "nvidia", "100a"),
    ("rtx5090", "nvidia", "120a"),
    ("mi300x", "amdgpu", "94"),
]
