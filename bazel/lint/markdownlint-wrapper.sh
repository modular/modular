

set -euo pipefail

readonly binary=$(find $PWD -name markdownlint -path "*markdownlint_*")
readonly config=$(find $BUILD_WORKSPACE_DIRECTORY -name .markdownlint.yaml -path "*bazel/lint*")

JS_BINARY__CHDIR="$BUILD_WORKSPACE_DIRECTORY" \
  "$binary" --config "$config" \
  --ignore-path "$BUILD_WORKSPACE_DIRECTORY/.gitignore" \
  --ignore "$BUILD_WORKSPACE_DIRECTORY/third-party" \
  "$@" . 2>&1 | sed 's/^/error: /'
