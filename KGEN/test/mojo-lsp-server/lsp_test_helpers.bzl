"""Helpers for mojo-lsp-server test targets."""

def parse_entry(e):
    """Normalize a plain string or (subdir, suffix, regex) tuple to a dict.

    Args:
        e: Either a plain subdir string or a (subdir, suffix, regex) tuple.
            The suffix is used as the Bazel target name suffix.
            The regex (if non-None) is passed to --file-regex.

    Returns:
        A dict with keys "subdir", "suffix", and "regex".
    """
    if type(e) == "string":
        return {"subdir": e, "suffix": e, "regex": None}
    subdir, suffix, regex = e
    return {"subdir": subdir, "suffix": suffix, "regex": regex}
