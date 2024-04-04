# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   src/black/const.py
#
# ===----------------------------------------------------------------------=== #

DEFAULT_LINE_LENGTH = 80
DEFAULT_EXCLUDES = r"/(\.direnv|\.eggs|\.git|\.hg|\.mypy_cache|\.nox|\.tox|\.venv|venv|\.svn|\.ipynb_checkpoints|_build|buck-out|build|dist|__pypackages__)/"  # noqa: B950
DEFAULT_INCLUDES = r"(\.pyi?|\.ipynb)$"
STDIN_PLACEHOLDER = "__BLACK_STDIN_FILENAME__"
