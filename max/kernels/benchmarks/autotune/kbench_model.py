# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from __future__ import annotations

import copy
import csv
import functools
import glob
import logging
import math
import os
import shutil
import string
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from subprocess import list2cmdline
from time import time
from typing import Any

import numpy as np
import pandas as pd
import utils
import yaml
from rich.progress import (
    Progress,
)


@dataclass
class ProcessOutput:
    stdout: str | None = None
    stderr: str | None = None
    return_code: int = -1
    path: Path | None = None

    def log(self) -> None:
        if self.stdout:
            logging.debug("output " + self.stdout + utils.LINE)
        if self.stderr:
            logging.debug("error " + self.stderr + utils.LINE)


# TODO: remove and replace directly with subprocess.run
def _run_cmdline(
    cmd: list[str],
    dryrun: bool = False,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> ProcessOutput:
    """Execute a shell command with error handling."""
    if env is None:
        env = {}
    try:
        if dryrun:
            print(list2cmdline(cmd))
            return ProcessOutput(None, None, -1, None)

        # Pass the current environment to subprocess, including MODULAR_MOJO_MAX_IMPORT_PATH
        _env = os.environ.copy()
        _env.update(dict(env))
        if timeout is None:
            output = subprocess.run(
                cmd, check=False, capture_output=True, env=_env
            )
        else:
            try:
                output = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    env=_env,
                    timeout=timeout,
                )
            except Exception as e:
                return ProcessOutput(None, str(e), os.EX_OSERR)

        return ProcessOutput(
            output.stdout.decode("utf-8"),
            output.stderr.decode("utf-8"),
            output.returncode,
        )

    except Exception as exc:
        return ProcessOutput(
            None,
            f"Unable to run command {list2cmdline(cmd)}: {exc}",
            os.EX_OSERR,
        )


@dataclass(frozen=True)
class Lang:
    name: str
    extensions: list[str]
    path: str
    needs_compilation: bool


# TODO: enabled cached property option
# @functools.cached_property
# @staticmethod
def mojo_binary() -> str:
    """Find mojo binary in PATH."""
    # Check for Bazel-provided mojo binary first
    if mojo_path := os.environ.get("MODULAR_MOJO_MAX_DRIVER_PATH"):
        if os.path.exists(mojo_path):
            return mojo_path
        else:
            raise FileNotFoundError(
                f"MODULAR_MOJO_MAX_DRIVER_PATH '{mojo_path}' does not exist."
            )
    # Fall back to searching in PATH
    if mojo := shutil.which("mojo"):
        return mojo
    raise FileNotFoundError("Could not find the `mojo` binary.")


def python_binary() -> str:
    """Find python binary in PATH."""
    return sys.executable


class SupportedLangs:
    MOJO = Lang("mojo", [".mojo"], mojo_binary(), needs_compilation=True)
    PYTHON = Lang("python", [".py"], python_binary(), needs_compilation=False)

    @staticmethod
    def which_executor(file: Path) -> Lang:
        if file.suffix in SupportedLangs.PYTHON.extensions:
            return SupportedLangs.PYTHON
        elif file.suffix in SupportedLangs.MOJO.extensions:
            return SupportedLangs.MOJO
        else:
            raise ValueError(f"Extension {file.suffix} is not supported!")


@dataclass
class Param:
    name: str
    value: Any

    def define(self, lang: Lang) -> list[str]:
        """Generate command line arguments for this parameter."""

        if lang == SupportedLangs.MOJO:
            if self.name.startswith("$"):
                var_name = self.name.removeprefix("$")
                return [f"--{var_name}={self.value}"]
            return ["-D", f"{self.name}={self.value}"]
        if lang == SupportedLangs.PYTHON:
            var_name = self.name.removeprefix("$")
            return [f"--{var_name}={self.value}"]
        return [""]


@dataclass
class ParamSpace:
    name: str
    value: Any
    value_set: list[Any] = field(default_factory=list)
    length: int = 0

    def __post_init__(self) -> None:
        """Initialize value set from flattened values."""
        # Try evaluating value as an arithmetic expression:
        try:
            if not isinstance(self.value, list):
                self.value = [self.value]
            self.value = [eval(x) for x in self.value]
        except:
            pass
        # Note: as of python3.7+ the built-in dict is guaranteed to maintain insertion order.
        self.value_set = list(dict.fromkeys(utils.flatten(self.value)))
        self.value = None
        self.length = len(self.value_set)


# Singleton build failed state
@dataclass(frozen=True)
class _BuildFailed:
    pass


BuildFailed = _BuildFailed()


class KBENCH_MODE(Enum):
    RUN = auto()
    BUILD = auto()
    BUILD_AND_RUN = auto()


class KbenchCache:
    """Cache for compiled binaries."""

    def __init__(self, path: Path | str = "kbench_cache.pkl") -> None:
        self.path = Path(path)
        self.data: dict[str, str | _BuildFailed] = {}
        self.is_active = False

    def clear(self) -> None:
        """Remove cache file if it exists."""
        logging.debug(f"Removing kbench-cache: {self.path}")
        if self.path.exists():
            subprocess.run(["rm", str(self.path)])

    def load(self) -> None:
        """Load cache from file."""
        if self.path.exists():
            self.data = utils.load_pickle(self.path)
        self.is_active = True

    def dump(self) -> None:
        """Save cache to file."""
        if self.is_active and self.data:
            utils.store_pickle(self.path, self.data)

    def query(self, key: str) -> str | _BuildFailed | None:
        """Get cached path for given key if it exists."""
        if not self.is_active:
            return None
        obj_path = self.data.get(key)
        if isinstance(obj_path, str):
            return obj_path if Path(obj_path).exists() else None
        return obj_path

    def store(self, key: str, obj_path: Path) -> Path | None:
        """Store object in cache and return its new path."""
        if not self.is_active:
            return None
        # TODO: revise the following conflict.
        if key in self.data:
            logging.debug(f"overwriting {key} already in obj-cache")
        self.data[key] = str(obj_path)
        return obj_path

    def store_failed(self, key: str) -> None:
        """Store build failure result for the specified key."""
        if not self.is_active:
            return None
        # TODO: revise the following conflict.
        if key in self.data:
            logging.debug(f"overwriting {key} already in obj-cache")
        self.data[key] = BuildFailed


@dataclass(frozen=True)
class SpecInstance:
    name: str
    file: Path
    executor: Lang
    params: list[Param] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.params)

    @functools.cached_property
    def _get_defines(self) -> list[str]:
        defines = []
        for param in self.params:
            if not param.name.startswith("$"):
                defines.append(param.define(self.executor))

        return [item for sublist in defines for item in sublist]

    @functools.cached_property
    def _get_vars(self) -> list[str]:
        vars = []
        for param in self.params:
            if param.name.startswith("$"):
                vars.append(param.define(self.executor))

        return [item for sublist in vars for item in sublist]

    def build(
        self,
        *,
        output_dir: Path,
        build_opts: list[str] = [],  # noqa: B006
        dryrun: bool = False,
        idx: int = -1,
        enable_logging: bool = True,
    ) -> ProcessOutput:
        """Build the spec instance. Use set of compile-time
        parameters as path of the compiled binary and store
        the executable in 'output_dir'.
        """

        bin_name = self.hash(with_variables=False)
        bin_path = output_dir / Path(bin_name)

        if enable_logging:
            logging.info(f"building [{idx}][{bin_name}]")
            logging.debug(
                f"defines: {self._get_defines}"
                + "\n"
                + f"vars   : {self._get_vars}"
            )

        executor = self.executor

        if not executor.needs_compilation:
            return ProcessOutput(return_code=os.EX_OK, path=self.file)

        if executor == SupportedLangs.MOJO:
            cmd = [executor.path]
            cmd.extend(["build"])
            if build_opts:
                cmd.extend(build_opts)
            cmd.extend(
                [
                    *self._get_defines,
                    str(self.file),
                    "-o",
                    str(bin_path),
                ]
            )
            out = _run_cmdline(cmd, dryrun)
            if out.return_code == os.EX_OK:
                out.path = bin_path
            else:
                out.path = None
            return out

        return ProcessOutput()

    def execute(
        self,
        binary_path: Path,
        output_file: Path,
        dryrun: bool = False,
        exec_prefix: list[str] = [],  # noqa: B006
        exec_suffix: list[str] = [],  # noqa: B006
        env: dict[str, str] = {},  # noqa: B006
        timeout_secs: int | None = None,
    ) -> ProcessOutput:
        if self.executor == SupportedLangs.PYTHON:
            exec_prefix = exec_prefix + [self.executor.path]
            vars = self._get_defines + self._get_vars
        else:
            vars = self._get_vars

        cmd = []
        if exec_prefix:
            logging.debug(f"exec-prefix: {exec_prefix}")
            cmd.extend(exec_prefix)
        cmd.extend([str(binary_path), *vars, "-o", str(output_file)])
        if exec_suffix:
            cmd.extend(exec_suffix)
            logging.debug(f"exec-suffix: {exec_suffix}")
        out = _run_cmdline(cmd, dryrun, timeout=timeout_secs, env=env)
        return out

    def to_obj(self) -> dict[str, Any]:
        return {param.name: param.value for param in self.params}

    @functools.cached_property
    def file_stem(self) -> str:
        return Path(self.file).with_suffix("").stem

    def __str__(self) -> str:
        tokens = [self.file_stem]
        for param in self.params:
            tokens.append(f"{param.name}={param.value}")
        return "/".join(tokens)

    def hash(self, with_variables: bool = True) -> str:
        MAX_FILENAME_LEN = 224

        tokens = [self.file_stem]
        for param in self.params:
            name = param.name
            # just use compile-time parameters and ignore runtime variables.
            if name.startswith("$") and not with_variables:
                continue
            name = name.replace("$", "")
            tokens.append(f"{name}-{param.value}")

        hash_str = "_".join(tokens)
        if len(hash_str) < MAX_FILENAME_LEN:
            return hash_str
        else:
            MAX_HASH_DIGITS = 8
            hash_hex = hash(hash_str) % (10**MAX_HASH_DIGITS)
            return f"{hash_str[: MAX_FILENAME_LEN - MAX_HASH_DIGITS]}{hash_hex}"


class GridSearchStrategy:
    instances: list[SpecInstance] = field(default_factory=list)

    def __init__(self, name, file, params) -> None:  # noqa: ANN001
        self.instances: list[SpecInstance] = []

        # Expand the product of all the param:value-set's per each group of parameters
        for cfg in params:
            name_list = [p.name for p in cfg]
            param_list = [p.value_set for p in cfg]
            param_mesh = list(product(*param_list))
            num_params = len(cfg)
            for idx in range(len(param_mesh)):
                s = SpecInstance(
                    name=name,
                    file=file,
                    params=[
                        Param(name=name_list[i], value=param_mesh[idx][i])
                        for i in range(num_params)
                    ],
                    executor=SupportedLangs.which_executor(file),
                )
                self.instances.append(s)

    def __iter__(self):
        self.offset = 0
        return self

    def __next__(self):
        # Stop condition
        if self.offset == len(self.instances):
            raise StopIteration

        res = self.instances[self.offset]
        self.offset += 1
        return res

    def __getitem__(self, i):  # noqa: ANN001
        return self.instances[i]

    def __len__(self) -> int:
        return len(self.instances)

    def extend(self, other) -> None:  # noqa: ANN001
        self.instances.extend(other.instances)


@dataclass(repr=True)
class Spec:
    name: str = ""
    file: Path = Path("")
    params: list[list[ParamSpace]] = field(default_factory=list)
    mesh_idx: int = 0
    mesh: list[SpecInstance] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)

    @staticmethod
    def load_yaml(file: Path) -> Spec:
        """
        Loads the spec from a YAML file

        Args:
            file (Path): the yaml file Path

        Returns:
            Spec: the spec
        """
        if not file.exists():
            raise FileNotFoundError(
                f'Unable to find the spec file at "{file}".'
            )
        try:
            logging.info(f"Loading yaml [{file}]" + utils.LINE)
            return Spec.loads(file.read_text())
        except Exception as e:
            raise ValueError(f"Could not load spec from {file}\nException: {e}")  # noqa: B904

    @staticmethod
    def load_yaml_list(yaml_path_list: list[str]) -> Spec:
        spec: Spec = None  # type: ignore
        for i, yaml_path in enumerate(yaml_path_list):
            spec_ld = Spec.load_yaml(Path(yaml_path))
            if i == 0:
                spec = spec_ld
            else:
                spec.join(spec_ld)
        return spec

    @staticmethod
    def parse_params(param_list: list[str]):  # noqa: ANN205
        """
        Parse the parameters as (key,value) dictionary.
        The parameters can be defined as follows:
        - `PARAM_NAME:PARAM_VALUE` (single value)
        - `PARAM_NAME:[PARAM_VALUE0, PARAM_VALUE1]` (Pythonic list of values)

        Args:
            param_list (List): a list of param-value's as strings/

        Returns:
            Spec: Dictionary of with extra param names as keys and param values.
        """
        d: dict[str, list] = {}
        IFS = ":"
        for p in param_list:
            name = ""
            val = ""
            if IFS in p:
                name, val = p.split(IFS)

            if name not in d:
                d[name] = []

            # This supports list of params per one definition
            # The following works for parsing a single-value, or a Pythonic list of values.
            vals = val.split(",")
            vals[0] = vals[0].strip("[")
            vals[-1] = vals[-1].strip("]")
            for i, v in enumerate(vals):
                v = v.strip()
                try:
                    vals[i] = eval(v)
                except:
                    vals[i] = v
            d[name].extend(vals)
        return d

    def extend_params(self, param_list: list[str]) -> None:
        # Expand with CLI params
        extra_params = self.parse_params(param_list)

        # For all params in each config either, update the existing `value_set`
        # with the new param value(s).
        for cfg in self.params:
            for k, v in extra_params.items():
                found = False
                for ps in cfg:
                    if ps.name == k:
                        ps.value_set.append(v)
                        ps.value_set = list(
                            dict.fromkeys(utils.flatten(ps.value_set))
                        )
                        found = True
                        break
                if not found:
                    cfg.append(ParamSpace(k, v))

        self.setup_mesh()

    def extend_shape_params(self, param_set: list[Param]) -> None:
        # TODO: check for collisions in param-names

        extra_params: list[ParamSpace] = []
        for ps in param_set:
            extra_params.append(ParamSpace(ps.name, ps.value))

        # add extended set of parameter to each bundle of parameters:
        for p in self.params:
            p.extend(extra_params)

        if not self.params:
            self.params = [extra_params]
        self.setup_mesh()

    def dump_yaml(self, out_path: Path) -> None:
        assert self.mesh, "There are no instances to write to YAML!"
        obj = {
            "name": self.name,
            "file": self.file,
            "params": [s.to_obj() for s in self.mesh],
        }
        with open(out_path, "w") as f:
            yaml.dump(obj, f, sort_keys=False)
        logging.debug(f"dumped {len(self.mesh)} instances to [{out_path}]")

    @staticmethod
    def loads(yaml_str: str) -> Spec:
        """
        Deserializes a Spec object from the given yaml string.

        Args:
            yaml_str (str): the yaml string representation of the model manifest

        Returns:
            Spec: a Spec loaded from the given yaml string
        """
        obj = yaml.safe_load(yaml_str)

        if "name" not in obj:
            logging.warning("Field [name] is not set in YAML")
        if "file" not in obj:
            logging.warning("Field [file] is not set in YAML")

        params: list[list[ParamSpace]] = []
        if "params" in obj:
            for cfg in obj["params"]:
                e: list[ParamSpace] = []
                for k, v in cfg.items():
                    if k == "metadata":
                        continue
                    e.append(ParamSpace(name=k, value=v))
                params.append(e)

        return Spec(
            name=obj.get("name", ""),
            file=obj.get("file", ""),
            params=params,
            rules=obj.get("rules", []),
        )

    def __len__(self) -> int:
        return len(self.mesh)

    def __post_init__(self):
        # checking if the file source path is valid
        file_abs_path = Path(
            string.Template(str(self.file)).substitute(os.environ)
        ).absolute()
        assert file_abs_path.exists(), (
            f"error: '{file_abs_path}' does not exist."
        )
        self.file = file_abs_path

        # setup mesh
        if self.params:
            self.setup_mesh()
        else:
            # default values for empty mesh
            self.mesh = [
                SpecInstance("", Path("./"), executor=SupportedLangs.MOJO)
            ]

    def setup_mesh(self):  # noqa: ANN201
        """
        Setup a mesh (cartesian product) of all values for all params. For example,
        if we have 2 set of params M=[64,256] and N=[A,B,C], the mesh will include
        to the following values:

        M=[64,256] x N=[A,B,C]
        ======================
        idx  : values
        0    : [64,A]
        1    : [64,B]
        2    : [64,C]
        3    : [256,A]
        4    : [256,B]
        5    : [256,C]

        At the end, append the configs with fixed parameters, if any exists in YAML.

        Return the total size of mesh.
        """
        grid_mesh = list(GridSearchStrategy(self.name, self.file, self.params))
        self.mesh = self.apply_rules(grid_mesh, self.rules)
        return len(self.mesh)

    def join(self, other: Spec) -> None:
        assert self.name == other.name
        assert self.file == other.file
        assert len(other.mesh) > 0

        self.mesh_idx = 0
        self.params.extend(other.params)
        self.mesh.extend(other.mesh)

    @staticmethod
    def apply_rules(
        mesh: list[SpecInstance], rules: list[str]
    ) -> list[SpecInstance]:
        new_mesh: list[SpecInstance] = []

        if not rules:
            return mesh

        def remove_dlr(s: str) -> str:
            return s.replace("$", "")

        for s in mesh:
            valid = True
            for r in rules:
                # TODO: revise handling of $ in string.
                locals = {remove_dlr(p.name): p.value for p in s.params}
                r = remove_dlr(r)

                try:
                    e = eval(r, {}, locals)
                # the following exception is required in case a parameter
                # is present in rule and missing from spec-instance combination.
                except NameError:
                    e = True
                valid = valid & e
                if not valid:
                    break
            if valid:
                new_mesh.append(s)
        return new_mesh

    def filter(self, filter_list: list[str]) -> None:
        filters: dict[str, list] = {}
        for f in filter_list:
            if "=" in f:
                name, val = f.split("=")
            elif ":" in f:
                name, val = f.split(":")

            if name not in filters:
                filters[name] = []
            filters[name].append(val)

        filtered_insts: list[SpecInstance] = []
        num_filters = len(filter_list)

        # Count the number of valid filters in each instance.
        # If the count==num_filters then add the instance to the result.
        valid_cnt = np.zeros(len(self.mesh), dtype=np.int32)

        for k_filter, v_filter in filters.items():
            for i, s in enumerate(self.mesh):
                for p in s.params:
                    if p.name == k_filter and str(p.value) in v_filter:
                        valid_cnt[i] += 1

        for i, idx in enumerate(valid_cnt):
            if idx == num_filters:
                filtered_insts.append(self.mesh[i])

        self.mesh = filtered_insts[:]
        self.mesh_idx = 0

    def __iter__(self):
        self.iter_offset = 0
        return self

    def __next__(self) -> SpecInstance:
        assert self.mesh is not None, (
            "Should call self.init_mesh after loading or in postinit."
        )

        # Stop condition
        if self.iter_offset == len(self.mesh):
            raise StopIteration

        # Retrieve and update self.mesh_idx
        idx = self.iter_offset
        self.iter_offset += 1
        return self.mesh[idx]

    def __repr__(self) -> str:
        rs = [f"[{i}] {str(s)}" for i, s in enumerate(self.mesh)]
        rs += [utils.LINE]
        rs += [f"Num Instances: {len(self.mesh)}"]
        rs += [utils.LINE]
        return "\n".join(rs)


@dataclass
class BuildItem:
    """
    To store all necessary details for building a spec item (instance).

    Args:
        idx: unique index of item in the list of scheduler items
        spec_instance: the parameter set used as the basis of build
        output_dir: output directory specific for this build item
        dryrun: set to True to enable dryrun
        output_path: path to output file
        bin_path: path to executable binary
        build_output: output message for build
        build_elapsed_time: elapsed time for build
        exec_output: output message for exec
        exec_benchmark_time: measured time for executing the entire benchmark
    """

    idx: int
    spec_instance: SpecInstance
    output_dir: Path
    build_opts: list
    dryrun: bool = False
    output_path: Path = Path()
    bin_path: Path | None = None

    build_output: ProcessOutput = field(default_factory=ProcessOutput)
    build_elapsed_time: float = 0
    exec_output: ProcessOutput = field(default_factory=ProcessOutput)
    exec_benchmark_time: float = 0


def _get_similar_files(path: Path) -> list[Path]:
    """Returns a list of files that belong to the same benchmark but are
    created by different processes, e.g. due to using mpirun
    """
    dir_name = os.path.dirname(path)
    stem = path.stem
    suffix = path.suffix
    pattern = os.path.join(dir_name, f"{stem}*{suffix}")
    return [Path(p) for p in sorted(glob.glob(pattern))]


class Scheduler:
    """
    Kbench singleton scheduler class to coordinate building and running all items in spec.

    Args:
        num_cpu: number of cpu's (cores) used for building items
        num_gpu: number of gpu's used for executing items
        build_items: list of spec items to build (BuildItem's)
        obj_cache: kbench obj-cache
        output_dir: parent output directory for all results
        num_specs: total number of spec items added to scheduler (to build+run)
    """

    num_cpu: int
    num_gpu: int
    build_items: list[BuildItem]
    obj_cache: KbenchCache
    run_only: bool
    output_suffix: str
    output_dir: Path
    num_specs: int
    num_unique_build_items: int = 0

    CHUNK_SIZE: int = 1
    EXEC_STRIDE: int = 100

    def __init__(
        self,
        num_cpu: int,
        num_gpu: int,
        obj_cache: KbenchCache,
        run_only: bool,
        spec_list: list[SpecInstance],
        output_dir: Path,
        build_opts: list[str],
        dryrun: bool,
        output_suffix: str = "output.csv",
        progress: Progress = Progress(),
    ) -> None:
        self.num_cpu = num_cpu
        self.num_gpu = num_gpu
        if not (0 < num_gpu <= num_cpu):
            raise ValueError(
                "num_gpu must be greater than 0 and less than or equal to num_cpu."
            )

        self.obj_cache = obj_cache
        self.num_specs = len(spec_list)
        output_dir_list = [
            Path(f"{output_dir}/out_{i}") for i in range(self.num_specs)
        ]
        self.output_suffix = output_suffix
        self.output_dir = output_dir
        self.run_only = run_only

        self.build_items = [
            BuildItem(
                idx=i,
                spec_instance=spec_list[i],
                output_dir=output_dir_list[i],
                build_opts=build_opts,
                dryrun=dryrun,
                output_path=output_dir_list[i] / output_suffix,
            )
            for i in range(self.num_specs)
        ]

        self.setup_build_pool()
        self.mk_output_dirs()
        self.progress = progress

    @staticmethod
    def kbench_mkdir(args: tuple[Path, str, bool]) -> Path:
        """Run the following command:
        `mkdir -p {output_dir}`
        """

        output_dir, output_suffix, run_only = args
        path_exists: bool = os.path.exists(output_dir) and os.path.isdir(
            output_dir
        )
        if not run_only:
            if path_exists:
                logging.warning(
                    f"Following output dir already exists and will be overwritten!\n[{str(output_dir)}]\n"
                )
                # Check for existing output files and remove them (if any):
                existing_csv = _get_similar_files(output_dir / output_suffix)
                for f in existing_csv:
                    os.remove(f)

            os.makedirs(output_dir, exist_ok=True)
        else:
            if not path_exists:
                raise ValueError(
                    f"--run-only specified but output directory does not exist: {output_dir}"
                )
        return output_dir

    def get_chunksize(self, num_elements: int) -> int:
        elements_per_cpu = math.ceil(num_elements / self.num_cpu)
        return min(elements_per_cpu, self.CHUNK_SIZE)

    def mk_output_dirs(self) -> None:
        """
        Make output directories for kbench results (one per spec-instance)
        """
        output_dir_list = [
            (b.output_dir, self.output_suffix, self.run_only)
            for b in self.build_items
        ]

        for r in self.build_pool.imap(
            self.kbench_mkdir,
            output_dir_list,
            chunksize=self.CHUNK_SIZE,
        ):
            logging.debug(f"mkdir [{r}]")
        logging.debug(
            "Created directories for all instances in spec." + utils.LINE
        )

    def schedule_unique_build_items(self) -> list[dict]:
        # Stores items that need to be build (i.e. not in cache)
        unique_build_items: dict[str, int] = {}
        # Stores paths to real binaries that have been cached beforehand
        unique_build_paths: dict[str, str] = {}

        for b in self.build_items:
            i = b.idx
            s = b.spec_instance
            bin_name = s.hash(with_variables=False)
            logging.debug(f"schedule [{i}][{bin_name}]")
            debug_msg = [
                f"defines: {s._get_defines}",
                f"vars   : {s._get_vars}",
            ]

            # first, check cache for build from previous rounds
            bin_path = self.obj_cache.query(bin_name)
            debug_msg += [f"In cache: {bool(bin_path)}"]
            if isinstance(bin_path, str):
                unique_build_paths[bin_name] = bin_path
            elif bin_path is BuildFailed:
                # This binary failed to build before and would just fail again.
                # Skip it.
                continue
            else:
                # Neither found in the cache, nor exists in the unique_build_items
                if bin_name not in unique_build_items:
                    unique_build_items[bin_name] = i
                    debug_msg += [f"Added to schedule (ref_idx=[{i}])"]
                else:
                    # Already in the unique_build_items list
                    idx = unique_build_items[bin_name]
                    debug_msg += [f"Currently in schedule (ref_idx=[{idx}])"]
            logging.debug("\n".join(debug_msg) + utils.LINE)
        return [unique_build_items, unique_build_paths]

    @staticmethod
    def _pool_build_wrapper(bi: BuildItem) -> BuildItem:
        t_start_item = time()
        bi.build_output = bi.spec_instance.build(
            output_dir=bi.output_dir,
            build_opts=bi.build_opts,
            dryrun=bi.dryrun,
            idx=bi.idx,
            enable_logging=False,
        )
        build_elapsed_time = int((time() - t_start_item) * 1e3)

        bi.build_elapsed_time = build_elapsed_time
        return bi

    def build_all(self) -> None:
        """
        Build all unique items scheduled by the scheduler.
        """

        unique_build_items_dict, unique_build_paths = (
            self.schedule_unique_build_items()
        )
        self.num_unique_build_items = len(unique_build_items_dict)

        if self.run_only and len(unique_build_items_dict) > 0:
            logging.error("Run only but not all binaries are found")
            raise ValueError(
                f"--run-only specified but {len(unique_build_items_dict)} binaries not found in cache. "
                "Please build first or remove --run-only flag."
            )

        unique_build_items = [
            self.build_items[i] for i in list(unique_build_items_dict.values())
        ]

        logging.info(
            f"scheduled {len(unique_build_items)} unique build items out of {self.num_specs}"
            + utils.LINE
        )
        if unique_build_items:
            obj_cache = self.obj_cache

            build_progress = self.progress.add_task(
                "build",
                total=len(unique_build_items),
                auto_refresh=False,
            )

            for cnt, b in enumerate(
                self.build_pool.imap(
                    self._pool_build_wrapper,
                    unique_build_items,
                    chunksize=self.CHUNK_SIZE,
                    # alternatively: self.get_chunksize(len(unique_build_items))
                )
            ):
                build_output = b.build_output
                # update the data with build_output result
                self.build_items[b.idx].build_output = build_output
                self.build_items[
                    b.idx
                ].build_elapsed_time = b.build_elapsed_time

                bin_name = b.spec_instance.hash(with_variables=False)

                num_unique_build_items = len(unique_build_items)
                logging.info(
                    f"build [{b.idx}][{bin_name}] ({utils._percentage(cnt + 1, num_unique_build_items)}%)"
                )

                # print build_output stdout and stderr using log function.
                build_output.log()

                # Try storing the executable in cache if:
                # - cache is active
                # - no error is reported in stderr
                # - build_output path is found
                if build_output.return_code == os.EX_OK and build_output.path:
                    binary_path = build_output.path
                    obj_cache.store(bin_name, binary_path)
                    unique_build_paths[bin_name] = binary_path
                else:
                    obj_cache.store_failed(bin_name)

                self.progress.update(build_progress, advance=1)
            logging.info(
                f"finished building {self.num_unique_build_items} unique items"
                + utils.LINE
            )

        self.close_build_pool()

        # update all build items with their binary path
        for b in self.build_items:
            bin_name = b.spec_instance.hash(with_variables=False)
            self.build_items[b.idx].bin_path = unique_build_paths.get(bin_name)

    @staticmethod
    def execute_item(
        build_item: BuildItem,
        profile,  # noqa: ANN001
        exec_prefix,  # noqa: ANN001
        exec_suffix,  # noqa: ANN001
        env: dict[str, str] | None = None,
        timeout_secs: int | None = None,
    ) -> BuildItem:
        """Execute all the items in the scheduler"""

        if env is None:
            env = {}
        bin_name = build_item.spec_instance.hash(with_variables=False)

        exec_prefix_item = copy.deepcopy(exec_prefix)
        exec_suffix_item = copy.deepcopy(exec_suffix)
        env_item = copy.deepcopy(env)

        profile_output = f"{build_item.output_dir}/{bin_name}_profile"
        if profile in ["ncu", "ncu-single"]:
            exec_prefix_item.extend(["ncu", "-o", profile_output])
            if profile == "ncu-single":
                exec_suffix_item.extend(
                    ["--bench-max-iters=0", "--bench-max-batch-size=1"]
                )
        if profile in ["rocm", "rocprof-compute"]:
            exec_prefix_item.extend(
                f"rocprof-compute profile --name NAME -p {profile_output} --".split()
            )
            logging.info(f"writing profiling results to {profile_output}")

        if build_item.bin_path:
            t_start = time()
            exec_output = build_item.spec_instance.execute(
                build_item.bin_path,
                build_item.output_path,
                dryrun=build_item.dryrun,
                exec_prefix=exec_prefix_item,
                exec_suffix=exec_suffix_item,
                env=env_item,
                timeout_secs=timeout_secs,
            )
            build_item.exec_output = exec_output
            build_item.exec_benchmark_time = time() - t_start
            exec_output.log()
        else:
            logging.error(f"Could not find binary [{bin_name}]")

        return build_item

    @staticmethod
    def _pool_execute_item_wrapper(
        args: tuple[
            BuildItem, str, list[str], list[str], dict[str, str], int | None
        ],
    ) -> BuildItem:
        return Scheduler.execute_item(*args)

    def setup_build_pool(self) -> None:
        self.build_pool = Pool(self.num_cpu)

    def setup_execution_pool(self) -> None:
        self.execution_pool = Pool(self.num_gpu)

    def close_build_pool(self) -> None:
        self.build_pool.close()
        self.build_pool.join()

    def close_execution_pool(self) -> None:
        self.execution_pool.close()
        self.execution_pool.join()

    @staticmethod
    def get_build_df(bi_list: list[BuildItem]) -> pd.DataFrame:
        build_df = pd.DataFrame(
            {
                "name": ["build" for b in bi_list],
                "spec": [f"{str(b.spec_instance)}" for b in bi_list],
            }
        )

        build_elapsed_time_list = [b.build_elapsed_time for b in bi_list]
        build_df.insert(
            len(build_df.columns),
            "met (ms)",
            pd.Series(build_elapsed_time_list),
        )
        build_df.insert(len(build_df.columns), "iters", 1)
        build_df.insert(
            len(build_df.columns),
            "mesh_idx",
            pd.Series([bi.idx for bi in bi_list]),
        )
        build_df["met (ms)"] = build_df["met (ms)"].fillna(0)

        build_df["name"] = build_df["name"].astype("string")
        build_df["spec"] = build_df["spec"].astype("string")
        build_df["met (ms)"] = build_df["met (ms)"].astype("float64")

        return pd.DataFrame(
            build_df.loc[:, ["mesh_idx", "name", "met (ms)", "iters", "spec"]]
        )

    @staticmethod
    def load_csv_to_pd(
        mesh_idx: int, current_spec: SpecInstance, files: list[Path]
    ) -> list[pd.DataFrame]:
        valid_specs: list[pd.DataFrame] = []
        for f in files:
            df = pd.read_csv(f, index_col=None, header=0)
            if not df.empty:
                df.insert(0, "mesh_idx", mesh_idx)
                df.insert(len(df.columns), "spec", str(current_spec))
                # If there are more than one entries in CSV then bencher
                # has added an extra column at the end of name with input_id.
                # TODO: This will create multiple rows with same mesh_idx.
                # Ensure this doesn't cause issues with 'kprofile' utilities.
                # TODO: Set an alternative index if input_id is missing.
                if len(df) > 1:
                    if df["name"].str.contains("/input:id").all():
                        raise ValueError(
                            "Detected multiple lines in output. All entries should have /input_id:"
                        )
                    id_column = df["name"].str.split("/input_id:").str[-1]
                    df["spec"] = (
                        df["spec"].astype(str) + "/input_id=" + id_column
                    )
                valid_specs.append(df)
        return valid_specs

    # Retrieve, sort, and pick top choices
    @staticmethod
    def get_valid_specs(bi_list: list[BuildItem], spec: Spec):  # noqa: ANN205
        valid_specs: list[pd.DataFrame] = []
        invalid_specs: list[int] = []

        for idx, b in enumerate(bi_list):
            valid = False
            files = _get_similar_files(b.output_path)

            if b.exec_output.return_code == os.EX_OK:
                if files:
                    current_valid_specs = Scheduler.load_csv_to_pd(
                        mesh_idx=b.idx,
                        current_spec=spec.mesh[b.idx],
                        files=files,
                    )
                    valid_specs.extend(current_valid_specs)
                    valid = len(current_valid_specs) > 0

                # TODO: is this case still needed? why should successful
                # build without output.csv be considered as valid result?

                if not valid:
                    df = pd.DataFrame().from_dict(
                        {
                            "mesh_idx": [b.idx],
                            "name": ["-"],
                            "met (ms)": [0],
                            "iters": [0],
                            "spec": [str(spec.mesh[b.idx])],
                        }
                    )
                    valid_specs.append(df)
                    valid = True

            if not valid:
                invalid_specs.append(idx)

        return valid_specs, invalid_specs

    @staticmethod
    def dump(
        bi_list: list[BuildItem],
        spec: Spec,
        output_path: Path = Path(),
        mode: KBENCH_MODE = KBENCH_MODE.BUILD_AND_RUN,
        t_build_total: float = 0.0,
        t_benchmark_total: float = 0.0,
        t_elapsed_total: float = 0.0,
        verbose: bool = False,
    ) -> None:
        output_lines = []
        output_dict: dict[str, Any] = {}

        build_df = Scheduler.get_build_df(bi_list)
        output_dict["build_df"] = build_df

        output_lines += [utils.LINE]
        output_lines += ["Build time stats:"]
        output_lines += [build_df.to_string(index=False)]

        output_lines += [utils.LINE]
        output_lines += [f"Running ['{spec.file}']"]

        ###############################
        valid_specs, invalid_specs = Scheduler.get_valid_specs(bi_list, spec)
        num_invalid_specs = len(invalid_specs)
        num_valid_specs = len(valid_specs)

        if num_invalid_specs:
            output_lines += [utils.LINE]
            output_lines += [
                f"Number of invalid specs: {num_invalid_specs} (out of {len(spec)})"
            ]

            for idx in invalid_specs:
                s = bi_list[idx].spec_instance
                build_output = bi_list[idx].build_output
                # check build failure
                if build_output.stdout or build_output.stderr:
                    output_lines += [utils.LINE]
                    output_lines += [f"mesh_idx: [{idx}][{s.to_obj()}]"]
                    if build_output.stdout:
                        output_lines.append(build_output.stdout)
                    if build_output.stderr:
                        output_lines.append(build_output.stderr)

        output_lines += [utils.LINE]
        output_lines += [
            f"Number of valid executed specs: {num_valid_specs} (out of {len(spec)})"
        ]

        if num_valid_specs:
            merged_df = pd.concat(valid_specs, axis=0, ignore_index=True)
            # Convert 'name' and 'spec' columns to pandas string
            merged_df["name"] = merged_df["name"].astype("string")
            merged_df["spec"] = merged_df["spec"].astype("string")

            ###############################
            # Get the name of column 2 (met (ms))
            output_dict["merged_df"] = merged_df

            met_col = str(merged_df.columns[2])
            output_lines += [merged_df.to_string(index=False)]
            output_lines += [utils.LINE]
            ###############################
        t_overhead = t_elapsed_total - (t_build_total + t_benchmark_total)
        timing_details = pd.DataFrame(
            {
                "Step": ["build", "benchmark", "kbench overhead", "TOTAL"],
                "Total (s)": [
                    t_build_total,
                    t_benchmark_total,
                    t_overhead,
                    t_elapsed_total,
                ],
            }
        ).round(3)
        timing_str = "Total elapsed time per step:\n" + str(
            timing_details.to_markdown(index=False, tablefmt="rounded_grid")
        )
        output_lines += [timing_str]
        output_str = "\n".join(output_lines)
        if verbose:
            print(output_str)
        else:
            logging.info(timing_str)

        if output_path:
            output_dict["name"] = spec.name
            output_dict["file"] = spec.file
            output_suffix = output_path.suffix
            pkl_path = output_path.with_suffix(output_suffix + ".pkl")
            csv_path = output_path.with_suffix(output_suffix + ".csv")
            txt_path = output_path.with_suffix(output_suffix + ".txt")

            utils.store_pickle(f"{pkl_path}", output_dict)

            # KBENCH_MODE.RUN overrides everything else and just dumps the running results.
            # THIS IS CRITICAL for CI automated kernel benchmarks workflow.
            if (
                mode in [KBENCH_MODE.RUN, KBENCH_MODE.BUILD_AND_RUN]
            ) and valid_specs:
                merged_df.drop(columns=["mesh_idx"]).to_csv(
                    csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC
                )
            elif mode == KBENCH_MODE.BUILD:
                build_df.to_csv(
                    csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC
                )

            with open(txt_path, "w") as f:
                f.write(output_str + "\n")
            logging.info(f"wrote results to [{txt_path}]")
            logging.info(f"wrote results to [{csv_path}]")
            logging.info(f"wrote results to [{pkl_path}]")
