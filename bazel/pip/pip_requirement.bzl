"""Wrapper around rules_python's requirement function to handle platform selection."""

def pip_requirement(name):
    return "@modular_pip_lock_file_repo//deps:{}".format(name)
