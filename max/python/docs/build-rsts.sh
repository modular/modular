#!/bin/bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##===----------------------------------------------------------------------===##

# Pass this script a directory name with .py files to generate
# a corresponding directory of .rst files. For example:
#   cd max/python
#   bash docs/build-rsts.sh max/nn
# There are some things this script does not currently do:
#   - Specify a `slug` front-matter for pages like attention/attention.rst
#   - List the `code_link` names on the index.rst file
#   - List the sub-package names in the toctree

if [ -z "$1" ]; then
    echo "Usage: $0 <source_path>"
    exit 1
fi

SOURCE_DIR="$1"

# The given path must start with "max/"
if [[ "$SOURCE_DIR" != max/* ]]; then
    echo "Error: The source path must start with 'max/'"
    exit 1
fi

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
# Remove the "max/" prefix from SOURCE_DIR
TARGET_DIR="$SCRIPT_DIR/${SOURCE_DIR#max/}"

# Convert SOURCE_DIR to package format (replace / with . and trim trailing slash)
PACKAGE_NAME="${SOURCE_DIR%/}"
PACKAGE_NAME="${PACKAGE_NAME//\//.}"

# Find all .py files and create corresponding .rst files
find "$SOURCE_DIR" -type f -name "*.py" | while read -r file; do
    # Get the relative path
    rel_path="${file#"$SOURCE_DIR"/}"
    # Get the directory of the file
    dir=$(dirname "$rel_path")
    # Get the filename without extension
    filename=$(basename "$file" .py)
    # Rename __init__.py to index.rst
    if [ "$filename" == "__init__" ]; then
        filename="index"
    fi
    # Determine title and type
    if [ "$filename" == "index" ]; then
        if [ "$dir" == "." ]; then
            title=$(basename "$SOURCE_DIR")  # Fix for the top-level index file
        else
            title=$(basename "$dir")
        fi
        type="package"
    else
        title="$filename"
        type="module"
    fi
    # Construct the new directory path
    new_dir="$TARGET_DIR/$dir"
    # Create the directory if it does not exist
    mkdir -p "$new_dir"
    # Construct the new .rst file path
    rst_file="$new_dir/$filename.rst"
    # Skip if file already exists
    if [ -f "$rst_file" ]; then
        echo "File already exists: $rst_file"
        continue
    fi
    # Write the front matter into the .rst file
    echo -e ":title: $title\n:type: $type\n:lang: python" > "$rst_file"
    # If filename is index, add extra front matter and title
    if [ "$filename" == "index" ]; then
        {
            echo -e ":wrapper_class: rst-index"
            echo -e "\n$title\n$(printf '%.0s-' $(seq 1 "${#title}"))"
            echo -e "\n.. toctree::\n   :hidden:\n"
        } >> "$rst_file"
        # List all other .py files in the original directory (ignoring __init__.py)
        find "$SOURCE_DIR/$dir" -maxdepth 1 -type f -name "*.py" ! -name "__init__.py" | while read -r py_file; do
            py_filename=$(basename "$py_file" .py)
            echo "   $py_filename" >> "$rst_file"
        done
    else
        # Construct module name
        if [[ "$rel_path" != */* ]]; then
            module_name="$PACKAGE_NAME.$title"
        else
            module_name="$PACKAGE_NAME.${dir//\//.}.$title"
        fi
        module_name="${module_name%.}"  # Remove trailing dot if dir is empty
        echo -e "\n.. automodule:: $module_name\n   :members:\n   :undoc-members:" >> "$rst_file"
    fi

    echo "Created: $rst_file"
done
