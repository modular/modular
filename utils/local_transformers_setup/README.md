# Local Transformers Setup Scripts

These scripts allow you to use a local, editable copy of the HuggingFace
transformers library for debugging purposes.

## Usage

### Setup

Run the setup script from anywhere in the workspace:

```bash
bash utils/local_transformers_setup/setup_local_transformers.sh
```

This will:

1. Clone the transformers repo (matching the version in `uv.lock`) to `local_transformers/`
2. Configure bazel to use the local copy instead of the pip package
3. Allow you to add prints, modify code, etc. in `local_transformers/src/transformers/`

### Making Changes

Edit any files in `local_transformers/src/transformers/` to print or save
intermediate tensors for model debugging.

Changes will be picked up on the next bazel build.

### Cleanup

When done debugging, restore everything:

```bash
bash utils/local_transformers_setup/cleanup_local_transformers.sh
```

This will:

1. Restore the original `pip_requirement.bzl`
2. Remove transformers_local configuration from `MODULE.bazel`
3. Delete the `local_transformers` directory
