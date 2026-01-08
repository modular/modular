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

"""Tools for extracting dependency information from source files."""

import sys
from dataclasses import dataclass
from pathlib import Path

import libcst as cst
from libcst import helpers as cst_help
from libcst import metadata as cst_metadata
from python_module import PythonModule
from typing_extensions import override

# Most third party top-level imports are the same as the package name, but not all.
# Ideally these can be determined from bazel, but some of them cannot be, specifically
# ones that share a top-level namespace, like the google and otel ones.
_THIRD_PARTY_IMPORTS = {
    "google_auth": ["google.auth"],
    "google_cloud_bigquery": ["google.cloud.bigquery"],
    "grpcio": ["grpc"],
    "protobuf": ["google.protobuf"],
    "ipython": ["IPython"],
    "levenshtein": ["Levenshtein"],
    "opencv_python": ["cv2"],
    "opentelemetry_api": [
        "opentelemetry.attributes",
        "opentelemetry.baggage",
        "opentelemetry.context",
        "opentelemetry.environment_variables",
        "opentelemetry._events",
        "opentelemetry._logs",
        "opentelemetry.metrics",
        "opentelemetry.propagate",
        "opentelemetry.propagators",
        "opentelemetry.trace",
        "opentelemetry.util",
        "opentelemetry.version",
    ],
    "opentelemetry_exporter_otlp_proto_http": [
        "opentelemetry.exporter.otlp.proto.http"
    ],
    "opentelemetry_exporter_prometheus": ["opentelemetry.exporter.prometheus"],
    "opentelemetry_sdk": ["opentelemetry.sdk"],
    "pillow": ["PIL"],
    "python_json_logger": ["pythonjsonlogger"],
    "ruamel_yaml": ["ruamel.yaml"],
    "pyyaml": ["yaml"],
    "pyzmq": ["zmq"],
}


@dataclass(frozen=True)
class _ThirdPartyDep:
    label: str
    import_paths: list[str]


# Raw label name, short name, import path
def _process_third_party_deps(
    third_party_deps: set[str],
) -> list[_ThirdPartyDep]:
    results = list[_ThirdPartyDep]()
    for dep in third_party_deps:
        if dep.startswith("@@rules_pycross"):
            # Format: @@rules_pycross++lock_file+modular_pip_lock_file_repo//deps:pytest@8.2.2
            name = dep.split(":")[-1].split("@")[0].replace("-", "_")
            if imports := _THIRD_PARTY_IMPORTS.get(name):
                import_paths = imports
            else:
                import_paths = [name.replace("-", "_")]
            results.append(_ThirdPartyDep(dep, import_paths))
        elif dep == "@@rules_python+//python/runfiles:runfiles":
            import_paths = ["python.runfiles"]
            results.append(_ThirdPartyDep(dep, import_paths))
        else:
            raise ValueError(f"Unsupported pycross dep format: {dep}")

    return results


# Adapted from https://github.com/bazel-contrib/rules_pydeps/blob/1c3eae19c4cd4b854e91a6ea48e21666b08d7ecc/pydeps/private/py/source_files.py
class _ImportFinder(cst.BatchableMetadataProvider[str]):
    METADATA_DEPENDENCIES = (cst_metadata.PositionProvider,)

    def __init__(self) -> None:
        super().__init__()

    @override
    def visit_Import(self, node: cst.Import) -> None:
        for name in node.names:
            self.set_metadata(
                node, f"{cst_help.get_full_name_for_node_or_raise(name.name)}"
            )

    @override
    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if not isinstance(node.names, cst.ImportStar):
            for name in node.names:
                # We are importing multiple things from one module,
                # _most likely_ we only need to worry about one, since they
                # should all come from the same target.
                # In theory we can do `from max import dtype, graph, mlir, ...`,
                # but we never do this so don't worry about it.
                if node.module:
                    self.set_metadata(
                        node,
                        "." * len(node.relative)
                        + f"{cst_help.get_full_name_for_node_or_raise(node.module)}.{name.name.value}",
                    )

    @override
    def visit_Decorator(self, node: cst.Decorator) -> None:
        name = cst_help.get_full_name_for_node_or_raise(node.decorator)
        if name == "pytest.mark.asyncio":
            self.set_metadata(node, name)


def _get_imports_from_file(path: Path) -> set[PythonModule]:
    with open(path) as file:
        content = file.read()
    wrapper = cst.MetadataWrapper(cst.parse_module(content))
    return set(
        PythonModule(imp) for imp in wrapper.resolve(_ImportFinder).values()
    )


def _is_third_party_import(
    third_party_deps: list[_ThirdPartyDep], mod: PythonModule
) -> str | None:
    """Checks if an import path is from a third party dependency. Returns the label of the dependency if so, otherwise None."""
    for dep in third_party_deps:
        # We can't check just `.startswith(dep.import_path)` here because
        # some import paths are prefixes of others (e.g. `pydantic` vs `pydantic_settings`).
        for path in dep.import_paths:
            if mod._module.startswith(path + ".") or mod._module == path:
                return dep.label
    return None


def check_dependencies_against_imports(
    working_dir: Path,
    sources: set[Path],
    third_party_deps: set[str],
    ignore_extra_deps: set[str],
    ignore_unresolved_imports: set[str],
    deps_info: dict[PythonModule, str],
    no_src_deps: set[str],
) -> tuple[set[str], set[str]]:
    """
    Scans sources for imports and checks them against provided dependencies.

    Args:
        working_dir: Absolute path to the root of the source files.
        sources: Sources of the current target, relative to working_dir.
        third_party_deps: Set of third party dependency labels.
        ignore_extra_deps: Set of dependency labels to ignore if unused.
        ignore_unresolved_imports: Set of import paths to ignore if unresolved.
        deps_info: Mapping from imports to the dependencies they correspond to.
        no_src_deps: Set of dependency labels that do not have source files.
            These are always considered unused.
    Returns: A set of unresolved imports and a set of unused dependencies.
    """
    if not working_dir.is_absolute():
        raise ValueError(f"working_dir must be absolute, found {working_dir}")

    if any(src.is_absolute() for src in sources):
        raise ValueError(
            f"Some source files were provided with absolute paths, found: {sources}"
        )

    unresolved: set[str] = set()
    used_deps: set[str] = set()
    all_deps: set[str] = set(deps_info.values()) | third_party_deps

    processed_third_party_deps = _process_third_party_deps(third_party_deps)

    # all source files in the provided set are considered local
    # and we turn the files into a set of modules
    local = {PythonModule.from_path(src) for src in sources}

    for source in sources:
        # Skip parsing in these cases, we can only parse Python files
        if source.suffix == ".so" or source.suffix == ".mojo":
            continue

        results = _get_imports_from_file(working_dir / source)
        for mod in results:
            if str(mod) == "pytest.mark.asyncio":
                # Special case this. Normally pytest.* would resolve to `pytest`, but
                # here we need to check for `pytest-asyncio`.
                # It's also an odd special case because it's only used in decorators.
                for dep in processed_third_party_deps:
                    if "pytest-asyncio" in dep.label:
                        used_deps.add(dep.label)
                        break
                else:
                    unresolved.add("pytest.mark.asyncio")
                continue

            absolute_import = mod
            if mod._module.startswith("."):
                # Relative import, convert
                n_dots = 0
                root = source
                while mod._module[n_dots] == ".":
                    root = root.parent
                    n_dots += 1

                if str(root) == ".":
                    path = mod._module.lstrip(".")
                else:
                    path = (
                        str(root).replace("/", ".")
                        + "."
                        + mod._module.lstrip(".")
                    )
                absolute_import = PythonModule(path)

            # 1. Standard library imports
            if (
                absolute_import.root() == "__future__"
                or absolute_import.root() in sys.stdlib_module_names
                # In the standard library too, but private.
                # Maybe we shouldn't use this, but for the sake of this test it's valid.
                or absolute_import.root() == "_typeshed"
            ):
                continue

            # 2. Direct import from sources
            if absolute_import in local:
                continue

            # 3: Direct match as a dependency
            if label := deps_info.get(absolute_import):
                used_deps.add(label)
                continue

            if absolute_import.has_parent():
                # 4. Import something from within a local module
                if absolute_import.parent() in local:
                    continue

                # 5. Import something from a dependency module
                if label := deps_info.get(absolute_import.parent()):
                    used_deps.add(label)
                    continue

            # 6. Third party dep
            if label := _is_third_party_import(processed_third_party_deps, mod):
                used_deps.add(label)
                continue

            # 7. Manual ignore
            if (
                mod._module in ignore_unresolved_imports
                or mod.root() in ignore_unresolved_imports
            ):
                # Resolved as a manual ignore
                continue

            # If none of the above, unresolved
            unresolved.add(mod._module)

    return unresolved, all_deps - used_deps - ignore_extra_deps | no_src_deps
