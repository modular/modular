# ===----------------------------------------------------------------------=== #
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
# ===----------------------------------------------------------------------=== #

# See lit.bzl for how this is used

import json
import os
import platform
from pathlib import Path

from lit.llvm import llvm_config

# llvm_config must have already been setup by the file we're appended to
assert llvm_config
# Include the only API that allows us to fail if we misconfigure this
import shutil

from lit.llvm.subst import ToolSubst
from python.runfiles import runfiles

_R = runfiles.Create()

# llvm_config.config.substitutions = substitutions_before
for index, pair in enumerate(llvm_config.config.substitutions):
    path = pair[1]
    if not path:
        # Empty substitution
        continue

    basename = os.path.basename(path)
    if basename in {"FileCheck", "not", "count"}:
        parts = path.split(" ")
        llvm_config.config.substitutions[index] = (
            pair[0],
            " ".join(parts[:-1] + [basename]),
        )

    elif os.path.dirname(path) == "../+llvm_configure+llvm-project/llvm":
        replacement = basename
        if rf"%\b{replacement}" in pair[0]:
            replacement = shutil.which(replacement)
            assert replacement, f"missing binary: {path}"
        llvm_config.config.substitutions[index] = (pair[0], replacement)

available_features: list[str] = json.loads("""{available_features}""")
for feature in available_features:
    assert not feature.startswith("$"), f"invalid feature: {feature}"
    llvm_config.config.available_features.add(feature)


def _replace_substitution(
    literal_command,  # noqa: ANN001
    subst_key,  # noqa: ANN001
    command,  # noqa: ANN001
    insert_in_front,  # noqa: ANN001
) -> None:
    for index, pair in reversed(
        list(enumerate(llvm_config.config.substitutions))
    ):
        if pair[0] in (literal_command, subst_key):
            del llvm_config.config.substitutions[index]

    if insert_in_front:
        llvm_config.config.substitutions.insert(0, (subst_key, command))
    else:
        llvm_config.config.substitutions.append((subst_key, command))


# Substituted in lit.bzl
custom_substitutions: dict[str, str] = json.loads("""{custom_substitutions}""")
for substitution, command in custom_substitutions.items():
    if " " in command:
        tool_path, arguments_str = command.split(" ", 1)
        arguments = arguments_str.split(" ")
    else:
        tool_path = command
        arguments = []
        if os.path.exists(tool_path):
            tool_path = os.path.abspath(tool_path)

    try:
        relative_tool_path = _R.Rlocation(tool_path)
        if os.path.exists(relative_tool_path):
            tool_path = relative_tool_path
        else:
            # Substitution could be a path, or a system binary
            _replace_substitution(substitution, substitution, command, True)
            continue
    except ValueError:
        # Substitution is a non-normalized path, ex: ../foo
        _replace_substitution(substitution, substitution, command, True)
        continue

    processed_arguments = []
    for arg in arguments:
        try:
            arg_path = _R.Rlocation(arg)
            if os.path.exists(arg_path):
                processed_arguments.append(arg_path)
            else:
                processed_arguments.append(arg)
        except ValueError:
            # Substitution is a non-normalized path, ex: ../foo
            processed_arguments.append(arg)

    subst_key, tool_pipe, command = ToolSubst(
        substitution,
        unresolved="fatal",
        command=tool_path,
        extra_args=processed_arguments,
    ).resolve(llvm_config, [])
    _replace_substitution(substitution, subst_key, tool_pipe + command, True)

    # Add %foo substitution before so it's preferred if it matches
    if not substitution.startswith("%"):
        subst_key, tool_pipe, command = ToolSubst(
            f"%{substitution}",
            unresolved="fatal",
            command=tool_path,
            extra_args=processed_arguments,
        ).resolve(llvm_config, [])
        _replace_substitution(
            f"%{substitution}", subst_key, tool_pipe + command, True
        )

new_system_libs = []
for arg in os.environ.get("MODULAR_MOJO_MAX_SYSTEM_LIBS", "").split(","):
    if arg.startswith("--sysroot=external/"):
        arg = "--sysroot=" + os.path.abspath(
            os.path.join("..", arg[len("--sysroot=external/") :])
        )
    new_system_libs.append(arg)


# Forward modular specific env vars for tool discovery. Fix the paths if necessary
import_paths = []
for key in sorted(os.environ.keys()):
    value = os.environ[key]
    # TODO: Remove this exception, it breaks some test
    if not key.startswith("BASH_FUNC"):
        if key == "MODULAR_MOJO_IMPORT_SRCS":
            # Bazel $(location :filegroup) gives us the path to a specific mojo
            # file, when what we want is 2 levels up. ex:
            # Kernels/mojo/stdlib/__init__.mojo -> Kernels/mojo. The mojo
            # searches that directory for the name 'stdlib'
            files = [Path(x) for x in value.replace(":", " ").split(" ")]
            import_paths.extend(
                x.parent.parent.absolute().as_posix() for x in files
            )
            continue
        elif key == "MODULAR_MOJO_MAX_IMPORT_PATH":
            import_paths.extend(
                Path(x).absolute().as_posix() for x in value.split(",")
            )
            continue
        elif key == "MODULAR_MOJO_MAX_COMPILERRT_PATH":
            # Convert to absolute path
            value = os.path.abspath(value)
            # Add rpath to system libs var so we can find it at runtime
            new_system_libs.extend(
                [
                    "-Xlinker",
                    "-rpath",
                    "-Xlinker",
                    os.path.dirname(value),
                ]
            )
        elif key == "MODULAR_MOJO_MAX_SYSTEM_LIBS":
            # NOTE: Handled manually above
            continue
        elif key == "MODULAR_MOJO_MAX_SHARED_LIBS":
            # Convert -Xlinker,-rpath,-Xlinker,relative/path/lib.so to absolute paths
            new_flags = []
            for flag in value.split(","):
                path = Path(flag).absolute()
                if path.exists():
                    new_flags.append(path.as_posix())
                else:
                    new_flags.append(flag)

            llvm_config.with_environment(key, ",".join(new_flags))
            continue
        elif key == "PYTHONPATH":
            # Add the toolchain's stdlib to the PYTHONPATH so nested actions
            # that run python use this toolchain instead of the system
            # toolchain, which may not exist or be compatible
            arch = "x86_64" if platform.machine() == "x86_64" else "aarch64"
            machine = (
                "apple-darwin"
                if platform.system() == "Darwin"
                else "unknown-linux-gnu"
            )
            major = platform.python_version_tuple()[0]
            minor = platform.python_version_tuple()[1]
            toolchain_path = (
                f"rules_python~~python~python_{major}_{minor}_{arch}-{machine}"
            )
            lib_path = f"{toolchain_path}/lib/python{major}.{minor}"
            bin_path = f"{toolchain_path}/bin"
            value = f"{value}:{_R.Rlocation(lib_path)}"

            # Make sure the toolchain's python3 is earlier in the PATH than any other python executable
            os.environ["PATH"] = (
                os.path.abspath(_R.Rlocation(bin_path))
                + ":"
                + os.environ["PATH"]
            )
            llvm_config.with_environment("PATH", os.environ["PATH"])
        elif key in ("ASAN_OPTIONS", "LSAN_OPTIONS"):
            new_options = []
            for option in value.split(","):
                option_key, option_value = option.split("=", 1)
                if os.path.exists(option_value):
                    option_value = os.path.abspath(option_value)
                new_options.append(f"{option_key}={option_value}")
            value = ",".join(new_options)
        elif os.path.exists(value):
            value = os.path.abspath(value)
        elif value.startswith("external/"):
            new_path = "../" + value[len("external/") :]
            if os.path.exists(new_path):
                value = os.path.abspath(new_path)

        llvm_config.with_environment(key, value)

if new_system_libs:
    llvm_config.with_environment(
        "MODULAR_MOJO_MAX_SYSTEM_LIBS", ",".join(new_system_libs)
    )

if import_paths:
    llvm_config.with_environment(
        "MODULAR_MOJO_MAX_IMPORT_PATH", ",".join(sorted(set(import_paths)))
    )
