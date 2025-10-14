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
from __future__ import annotations

import copy
import csv
import functools
import gc
import logging
import math
import os
import pickle
import shutil
import string
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from subprocess import list2cmdline
from time import time
from typing import Any

import click
import numpy as np
import pandas as pd
import rich
import yaml
from rich import print, traceback
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from utils import pretty_exception_handler

CONSOLE = Console(width=80)
CURRENT_FILE = Path(__file__).resolve()
LINE = "\n" + 70 * "-"


def store_pickle(path: Path | str, data: Any) -> None:
    """Serialize data to a pickle file."""
    with Path(path).open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path | str) -> Any:
    """Deserialize data from a pickle file."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def configure_logging(
    quiet: bool = False, verbose: bool = False, pretty_output: bool = True
) -> Console:
    """Configure logging with rich formatting."""
    global CONSOLE

    if pretty_output:
        debug_handler = RichHandler(
            show_path=False, show_time=False, console=CONSOLE
        )
        logging.basicConfig(format="%(message)s", handlers=[debug_handler])
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s")
        CONSOLE = Console(width=80, force_terminal=False, color_system=None)

    log_level = (
        logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    )
    logging.getLogger().setLevel(log_level)

    if verbose and pretty_output:
        traceback.install(suppress=[click, rich])
    elif pretty_output:
        sys.excepthook = pretty_exception_handler

    return CONSOLE


@dataclass
class Param:
    name: str
    value: Any

    def define(self) -> list[str]:
        """Generate command line arguments for this parameter."""
        if self.name.startswith("$"):
            var_name = self.name.split("$")[1]
            return [f"--{var_name}={self.value}"]
        return ["-D", f"{self.name}={self.value}"]


def flatten(value: int | object | Iterable) -> list[Any]:
    """Flatten nested iterables into a single list."""
    if not isinstance(value, Iterable) or isinstance(value, str):
        return [value]
    return [
        item
        for sublist in (flatten(item) for item in value)
        for item in sublist
    ]


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
        raise SystemExit(f"Unable to run command {list2cmdline(cmd)}") from exc


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
        self.value_set = list(dict.fromkeys(flatten(self.value)))
        self.value = None
        self.length = len(self.value_set)


@dataclass
class ProcessOutput:
    stdout: str | None = None
    stderr: str | None = None
    return_code: int = -1
    path: Path | None = None

    def log(self) -> None:
        if self.stdout:
            logging.debug("output " + self.stdout + LINE)
        if self.stderr:
            logging.debug("error " + self.stderr + LINE)


class KBENCH_MODE(Enum):
    RUN = auto()
    TUNE = auto()
    BUILD = auto()


class KbenchCache:
    """Cache for compiled binaries."""

    def __init__(self, path: Path | str = "kbench_cache.pkl") -> None:
        self.path = Path(path)
        self.data: dict[str, str] = {}
        self.is_active = False

    def clear(self) -> None:
        """Remove cache file if it exists."""
        logging.debug(f"Removing kbench-cache: {self.path}")
        if self.path.exists():
            subprocess.run(["rm", str(self.path)])

    def load(self) -> None:
        """Load cache from file."""
        if self.path.exists():
            self.data = load_pickle(self.path)
        self.is_active = True

    def dump(self) -> None:
        """Save cache to file."""
        if self.is_active and self.data:
            store_pickle(self.path, self.data)

    def query(self, key: str) -> str | None:
        """Get cached path for given key if it exists."""
        if not self.is_active:
            return None
        obj_path = str(self.data.get(key))
        return obj_path if Path(obj_path).exists() else None

    def store(self, key: str, obj_path: Path) -> Path | None:
        """Store object in cache and return its new path."""
        if not self.is_active:
            return None
        # TODO: revise the following conflict.
        if obj_path in self.data:
            logging.debug(f"overwriting {key} already in obj-cache")
        self.data[key] = str(obj_path)
        return obj_path


@dataclass(frozen=True)
class SpecInstance:
    name: str
    file: Path
    executor: str | None = None
    params: list[Param] = field(default_factory=list)

    @functools.cached_property
    def mojo_binary(self) -> str:
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

    def get_executor(self) -> list[str]:
        """Get executor command."""
        return self.executor.split() if self.executor else [self.mojo_binary]

    @functools.cached_property
    def _get_defines(self) -> list[str]:
        defines = []
        for param in self.params:
            if not param.name.startswith("$"):
                defines.append(param.define())

        return [item for sublist in defines for item in sublist]

    @functools.cached_property
    def _get_vars(self) -> list[str]:
        vars = []
        for param in self.params:
            if param.name.startswith("$"):
                vars.append(param.define())

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

        # TODO: add mojo-specific functions and allow for further expansion to other executors.
        cmd = self.get_executor()
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
            logging.info(f"Loading yaml [{file}]" + LINE)
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
                        ps.value_set = list(dict.fromkeys(ps.value_set))
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
            self.mesh = [SpecInstance("", Path("./"))]

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
        rs += [LINE]
        rs += [f"Num Instances: {len(self.mesh)}"]
        rs += [LINE]
        return "\n".join(rs)


def _get_tmp_path(file_path):  # noqa: ANN001, ANN202
    base = os.path.basename(file_path).split(".")[0]
    tf = tempfile.NamedTemporaryFile(prefix=str(base) + "_").name + "/"
    return Path(tf)


def _get_core_count():  # noqa: ANN202
    try:
        # The 'os.sched_getaffinity' method is only available on some Unix platforms
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined, unused-ignore]
    except AttributeError:
        # To cover other platforms, including mac
        return cpu_count()


def _get_visible_device_prefix(target_accelerator: str = "") -> str:
    if "nvidia" in target_accelerator or "cuda" in target_accelerator:
        return "CUDA_VISIBLE_DEVICES"
    elif "amd" in target_accelerator:
        return "ROCR_VISIBLE_DEVICES"
    else:
        return ""


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
    bin_path: Path = Path()

    build_output: ProcessOutput = field(default_factory=ProcessOutput)
    build_elapsed_time: float = 0
    exec_output: ProcessOutput = field(default_factory=ProcessOutput)
    exec_benchmark_time: float = 0


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
    output_dir: Path
    num_specs: int

    CHUNK_SIZE: int = 1
    EXEC_STRIDE: int = 100

    def __init__(
        self,
        num_cpu: int,
        num_gpu: int,
        obj_cache: KbenchCache,
        spec_list: list[SpecInstance],
        output_dir: Path,
        build_opts: list[str],
        dryrun: bool,
        output_suffix: str = "output.csv",
        progress: Progress = Progress(),
    ) -> None:
        self.num_cpu = num_cpu
        if not (0 < num_gpu <= num_cpu):
            raise ValueError(
                "num_gpu must be greater than 0 and less than or equal to num_cpu."
            )

        self.build_pool = Pool(num_cpu)
        self.execution_pool = Pool(num_gpu)
        self.obj_cache = obj_cache
        self.num_specs = len(spec_list)
        output_dir_list = [
            Path(f"{output_dir}/out_{i}") for i in range(self.num_specs)
        ]
        self.output_dir = output_dir

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

        self.mk_output_dirs()
        self.progress = progress

    @staticmethod
    def kbench_mkdir(output_dir):  # noqa: ANN001, ANN205
        """Run the following command:
        `rm -rf {output_dir} && mkdir -p {output_dir}`
        """
        # "rm -rf {output_dir} && mkdir -p {output_dir}"
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=False)
        return output_dir

    def get_chunksize(self, num_elements: int) -> int:
        elements_per_cpu = math.ceil(num_elements / self.num_cpu)
        return min(elements_per_cpu, self.CHUNK_SIZE)

    def mk_output_dirs(self) -> None:
        """
        Make output directories for kbench results (one per spec-instance)
        """
        output_dir_list = [b.output_dir for b in self.build_items]

        for r in self.build_pool.imap(
            self.kbench_mkdir,
            output_dir_list,
            chunksize=self.CHUNK_SIZE,
        ):
            logging.debug(f"mkdir [{r}]")
        logging.debug("Created directories for all instances in spec." + LINE)

    def schedule_unique_build_items(self) -> list[dict]:
        unique_build_items: dict[str, int] = {}
        unique_build_paths: dict[str, str] = {}
        for b in self.build_items:
            i = b.idx
            s = b.spec_instance
            bin_name = s.hash(with_variables=False)
            logging.info(f"schedule [{i}][{bin_name}]")
            debug_msg = [
                f"defines: {s._get_defines}",
                f"vars   : {s._get_vars}",
            ]

            # first, check cache for build from previous rounds
            bin_path = self.obj_cache.query(bin_name)
            debug_msg += [f"In cache: {bool(bin_path)}"]
            if bin_path:
                unique_build_paths[bin_name] = bin_path
            else:
                # Neither found in the cache, nor exists in the unique_build_items
                if bin_name not in unique_build_items:
                    unique_build_items[bin_name] = i
                    debug_msg += [f"Added to schedule (ref_idx=[{i}])"]
                else:
                    # Already in the unique_build_items list
                    idx = unique_build_items[bin_name]
                    debug_msg += [f"Currently in schedule (ref_idx=[{idx}])"]
            logging.debug("\n".join(debug_msg) + LINE)
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

    def build_all(self):  # noqa: ANN201
        """
        Build all unique items scheduled by the scheduler.
        """

        unique_build_items_dict, unique_build_paths = (
            self.schedule_unique_build_items()
        )
        unique_build_items = [
            self.build_items[i] for i in list(unique_build_items_dict.values())
        ]

        logging.info(
            f"scheduled {len(unique_build_items)} unique build items out of {self.num_specs}"
            + LINE
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

                self.progress.update(
                    build_progress,
                    description=f"build [ {b.idx} ][ {bin_name} ] finished",
                )

                num_unique_build_items = len(unique_build_items)
                logging.info(
                    f"build [{b.idx}][{bin_name}] ({_percentage(cnt + 1, num_unique_build_items)}%)"
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

                self.progress.update(build_progress, advance=1)
            logging.info(
                f"finished building {len(unique_build_paths)} unique items"
                + LINE
            )

        # update all build items with their binary path
        for b in self.build_items:
            bin_name = b.spec_instance.hash(with_variables=False)
            self.build_items[b.idx].bin_path = unique_build_paths.get(
                bin_name, Path()
            )

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

    # Retrieve, sort, and pick top choices
    @staticmethod
    def get_valid_specs(bi_list: list[BuildItem], spec: Spec):  # noqa: ANN205
        valid_specs: list[pd.DataFrame] = []
        invalid_specs: list[int] = []

        for idx, b in enumerate(bi_list):
            valid = False
            if os.path.exists(b.output_path):
                df = pd.read_csv(b.output_path, index_col=None, header=0)
                if not df.empty:
                    df.insert(0, "mesh_idx", b.idx)
                    df.insert(len(df.columns), "spec", str(spec.mesh[b.idx]))

                    valid_specs.append(df)
                    valid = True
            if not valid and b.exec_output.return_code == os.EX_OK:
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
        mode: KBENCH_MODE = KBENCH_MODE.RUN,
        t_build_total: float = 0.0,
        t_benchmark_total: float = 0.0,
        t_elapsed_total: float = 0.0,
        verbose: bool = False,
    ) -> None:
        output_lines = []
        output_dict: dict[str, Any] = {}

        build_df = Scheduler.get_build_df(bi_list)
        output_dict["build_df"] = build_df

        output_lines += [LINE]
        output_lines += ["Build time stats:"]
        output_lines += [build_df.to_string(index=False)]

        output_lines += [LINE]
        output_lines += [f"Running ['{spec.file}']"]

        ###############################
        valid_specs, invalid_specs = Scheduler.get_valid_specs(bi_list, spec)
        num_invalid_specs = len(invalid_specs)
        num_valid_specs = len(valid_specs)

        if num_invalid_specs:
            output_lines += [LINE]
            output_lines += [
                f"Number of invalid specs: {num_invalid_specs} (out of {len(spec)})"
            ]

            for idx in invalid_specs:
                s = bi_list[idx].spec_instance
                build_output = bi_list[idx].build_output
                # check build failure
                if build_output.stdout or build_output.stderr:
                    output_lines += [LINE]
                    output_lines += [f"mesh_idx: [{idx}][{s.to_obj()}]"]
                    if build_output.stdout:
                        output_lines.append(build_output.stdout)
                    if build_output.stderr:
                        output_lines.append(build_output.stderr)

        output_lines += [LINE]
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
            if mode == KBENCH_MODE.TUNE:
                tune_df = merged_df.sort_values([met_col], ascending=True)

                output_dict["tune_df"] = tune_df
                output_lines += [tune_df.to_string(index=False)]
                # Index to top spec after sort
                top_spec_idx = tune_df.iloc[0].mesh_idx
                runtime = tune_df.iloc[0][met_col]

                output_lines += [LINE]
                output_lines += ["Spec with the best measured time:"]
                output_lines += [
                    f"mesh_idx: {top_spec_idx} measured time: {runtime:6f} (ms)"
                ]
                output_lines += [bi_list[top_spec_idx].spec_instance.to_obj()]
                output_lines += [LINE]
            else:
                output_lines += [merged_df.to_string(index=False)]
                output_lines += [LINE]
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

            store_pickle(f"{pkl_path}", output_dict)

            # KBENCH_MODE.RUN overrides everything else and just dumps the running results.
            # THIS IS CRITICAL for CI automated kernel benchmarks workflow.
            if mode == KBENCH_MODE.RUN and valid_specs:
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


def _percentage(x: int, y: int) -> int:
    if x > 0 and y > 0:
        return int((x / y) * 100.0)
    return 0


def run(
    yaml_path_list,  # noqa: ANN001
    obj_cache: KbenchCache,
    shape: SpecInstance,
    output_path: Path = Path(),
    mode=KBENCH_MODE.RUN,  # noqa: ANN001
    param_list=None,  # noqa: ANN001
    filter_list=None,  # noqa: ANN001
    build_opts: list[str] = [],  # noqa: B006
    profile: str = "",
    exec_prefix: list[str] = [],  # noqa: B006
    exec_suffix: list[str] = [],  # noqa: B006
    dryrun: bool = False,
    verbose: bool = False,
    output_dir=None,  # noqa: ANN001
    num_cpu: int = 1,
    num_gpu: int = 1,
    target_accelerator: str | None = None,
    timeout_secs: int | None = None,
) -> None:
    if yaml_path_list:
        # Load specs from a list of YAML files and join them in 'spec'.
        assert len(yaml_path_list), "There should be at least 1 YAML as input."
        spec = Spec.load_yaml_list(yaml_path_list)
    else:
        # Just load an empty Spec with identical name and file as shape
        spec = Spec(shape.name, shape.file)

    if shape:
        spec.extend_shape_params(shape.params)

    # Expand with CLI params
    if param_list:
        spec.extend_params(param_list)

    # Apply the filters, if any.
    if filter_list:
        spec.filter(filter_list)

    if verbose:
        for i, s in enumerate(spec):
            logging.debug(f"[{i}]{s}")
        logging.debug(LINE)

    # Generate a tmp path for intermediate results.
    if not output_dir:
        output_dir = _get_tmp_path(spec.file)
    else:
        output_path = output_dir / output_path
    os.makedirs(output_path.parent, exist_ok=True)

    # strip output_path suffix
    if output_path.suffix in [".csv", ".pkl", ".txt"]:
        output_path = output_path.with_suffix("")

    output_dir = Path(output_dir)
    logging.info(f"output-dir: [{output_dir}]")

    # Run the code over the mesh of param/values
    t_start_total = time()
    progress = Progress(
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TextColumn("|"),
        TimeElapsedColumn(),
        console=CONSOLE,
        expand=True,
        transient=True,
    )

    # Set num_cpu to the half of maximum number of available CPUs
    if num_cpu == -1:
        num_cpu = max(_get_core_count() // 2, 1)

    logging.info(f"num cpu's: {num_cpu}")
    # Kbench Singleton Scheduler
    scheduler = Scheduler(
        num_cpu=num_cpu,
        num_gpu=num_gpu,
        obj_cache=obj_cache,
        spec_list=list(spec),
        output_dir=output_dir,
        build_opts=build_opts,
        dryrun=dryrun,
        output_suffix="output.csv",
        progress=progress,
    )

    visible_device_prefix = _get_visible_device_prefix(str(target_accelerator))

    # Run the code over the mesh of param/values
    t_start_total = time()
    t_benchmark_total = 0.0

    with progress:
        try:
            # Get the binary path for the unique list of build items
            # Build the binary if:
            # - could not find executable in the cache or cache is not active,
            # - could not find executable in the unique list of scheduled build items
            scheduler.build_all()
            scheduler.close_build_pool()
            obj_cache.dump()

            t_build_total = time() - t_start_total

            t_benchmark_start = time()
            if mode in [KBENCH_MODE.RUN, KBENCH_MODE.TUNE]:
                num_build_items = len(scheduler.build_items)
                exec_progress = scheduler.progress.add_task(
                    "run",
                    total=num_build_items,
                )

                # execute build items in batches of size Scheduler.EXEC_STRIDE
                for lower_bound in range(
                    0, num_build_items, Scheduler.EXEC_STRIDE
                ):
                    upper_bound = min(
                        lower_bound + Scheduler.EXEC_STRIDE, num_build_items
                    )

                    tasks = []
                    for cnt in range(lower_bound, upper_bound):
                        env = {}
                        if visible_device_prefix and num_gpu > 1:
                            env[visible_device_prefix] = str(cnt % num_gpu)

                        tasks.append(
                            (
                                scheduler.build_items[cnt],
                                profile,
                                exec_prefix,
                                exec_suffix,
                                env,
                                timeout_secs,
                            )
                        )

                    # This is to further ensure at most `num-gpu` processes are
                    # running at once, none of them sharing the same device context.
                    for cnt in range(lower_bound, upper_bound, num_gpu):
                        ub_gpu = min(upper_bound, cnt + num_gpu)
                        current_cnt = ub_gpu - cnt
                        for i, b in enumerate(
                            scheduler.execution_pool.imap(
                                scheduler._pool_execute_item_wrapper,
                                tasks[cnt - lower_bound : ub_gpu - lower_bound],
                                chunksize=1,
                            )
                        ):
                            scheduler.build_items[cnt + i] = b
                            logging.info(
                                f"running binary [{b.idx}/{num_build_items - 1}] ({_percentage(b.idx + 1, num_build_items)}%)"
                            )
                        scheduler.progress.update(
                            exec_progress, advance=current_cnt
                        )

                    t_benchmark_total = time() - t_benchmark_start
                    t_elapsed_total = time() - t_start_total

                    # dump results that have been executed so far
                    # ensure there are more than one iterations of stride loop.
                    if num_build_items >= Scheduler.EXEC_STRIDE:
                        Scheduler.dump(
                            scheduler.build_items[0:upper_bound],
                            spec,
                            output_path,
                            mode,
                            t_build_total,
                            t_benchmark_total,
                            t_elapsed_total,
                            verbose=False,
                        )
                logging.info("finished running all binaries")
                scheduler.close_execution_pool()

        except KeyboardInterrupt:
            scheduler.close_build_pool()
            scheduler.close_execution_pool()
            obj_cache.dump()
            sys.exit(0)

    ###############################
    # dump all the details
    t_elapsed_total = time() - t_start_total
    gc.collect()
    Scheduler.dump(
        scheduler.build_items,
        spec,
        output_path,
        mode,
        t_build_total,
        t_benchmark_total,
        t_elapsed_total,
        verbose=verbose,
    )
    logging.info(f"output-dir: [{output_dir}]\n{LINE}")


@functools.cache
def get_nvidia_smi():  # noqa: ANN201
    return shutil.which("nvidia-smi")


def reset_gpu() -> None:
    nvidia_smi = get_nvidia_smi()
    if not nvidia_smi:
        return
    subprocess.check_call([nvidia_smi, "-r"])


def check_gpu_clock() -> None:
    nvidia_smi = get_nvidia_smi()
    if not nvidia_smi:
        return
    output = subprocess.check_output(
        [
            nvidia_smi,
            "--query-gpu",
            "persistence_mode",
            "--format",
            "csv",
        ],
    )

    # We check for persistence here as a proxy to check if setup-gpu-clock.sh
    # has been run. This is not exact, but should cover most cases. Checking for
    # the clock frequency is more complicated since the frequencies changes per
    # GPU.
    if "Disabled" in output.decode("utf-8"):
        raise Exception(
            "the clock frequency for the GPU is not locked, please run"
            " `sudo utils/setup-gpu-clock.sh` to ensure the frequencies"
            " and power of the GPU are locked to get consistent"
            " benchmarking behavior."
        )


class FileGlobArg:
    def __init__(self, file: list[str]) -> None:
        self._files = file
        if not self._files:
            raise ValueError(
                f"Could not find any file that satisfies the glob {file}."
                " Check to ensure that the file exist or the glob matches one"
                " or more file."
            )

    def __iter__(self):
        return (Path(file).resolve() for file in self._files)

    def __len__(self) -> int:
        return len(self._files)


def set_build_opts(  # noqa: ANN201
    debug_level=None,  # noqa: ANN001
    optimization_level=None,  # noqa: ANN001
    use_experimental_kernels=None,  # noqa: ANN001
    target_accelerator=None,  # noqa: ANN001
    disable_warnings=None,  # noqa: ANN001
):
    build_opts = []
    if debug_level:
        build_opts.extend(["--debug-level", debug_level])
    if optimization_level:
        build_opts.extend([f"-O{optimization_level}"])
    if use_experimental_kernels:
        build_opts.extend(["-D", "USE_EXPERIMENTAL_KERNELS=1"])
    if target_accelerator:
        build_opts.extend(["--target-accelerator", target_accelerator])
    if disable_warnings:
        build_opts.extend(["--disable-warnings"])
    # TODO: add num_threads to CLI
    # num_threads_per_build = 1
    # build_opts.extend(["--num-threads", num_threads_per_build])
    return build_opts


def _validate_partition(partition: str) -> list[int]:
    assert ":" in partition
    partition_idx, num_partitions = [int(x) for x in partition.split(":")]
    assert num_partitions > 0
    assert 0 <= partition_idx < num_partitions, (
        "Condition: 0 <= partition_idx < num_partitions"
    )
    return [partition_idx, num_partitions]


target_accelerator_values = {
    "NVIDIA": [
        "nvidia:sm_52",
        "nvidia:sm_60",
        "nvidia:sm_61",
        "nvidia:sm_75",
        "nvidia:sm_80",
        "nvidia:sm_86",
        "nvidia:sm_87",
        "nvidia:sm_89",
        "nvidia:sm_90",
        "nvidia:sm_90a",
        "nvidia:sm_100",
        "nvidia:sm_100a",
        "nvidia:sm_120",
        "nvidia:sm_120a",
    ],
    "AMD": [
        "amdgpu:mi300x",
        "amdgpu:mi355x",
        "amdgpu:gfx942",
        "amdgpu:gfx950",
        "amdgpu:gfx1030",
        "amdgpu:gfx1100",
        "amdgpu:gfx1101",
        "amdgpu:gfx1102",
        "amdgpu:gfx1103",
        "amdgpu:gfx1150",
        "amdgpu:gfx1151",
        "amdgpu:gfx1152",
        "amdgpu:gfx1200",
        "amdgpu:gfx1201",
    ],
    "Apple": ["metal:1", "metal:2", "metal:3", "metal:4"],
}


@functools.cache
def get_target_accelerator_helpstr() -> str:
    helpstr = ""
    for arch, target_list in target_accelerator_values.items():
        helpstr += f"\n\n# {arch}\n\n"
        helpstr += ",\n".join([f"'{x}'" for x in target_list])
    return helpstr


help_str = "Benchmarking toolkit for Mojo kernels"


@click.command(help=help_str, no_args_is_help=True)
@click.option(
    "--filter",
    "filter",
    help=(
        "Define a single filter (should match a valid parameter, can have"
        " multiple ones). The filters should of the format `--filter"
        " PARAM=VALUE`, that is, the subset of parameters that satisfy this"
        " condition will be included."
    ),
    multiple=True,
)
@click.option(
    "--output",
    "-o",
    "output_path",
    default="output.csv",
    help="Path to output file.",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=None,
    help="Path to output directory for all results (default='/tmp')",
)
@click.option(
    "--tune",
    "-t",
    "tune",
    is_flag=True,
    default=False,
    help="Sort results by running time.",
)
@click.option(
    "--build",
    "build",
    is_flag=True,
    default=False,
    help="Just build the binary and report the build time.",
)
@click.option(
    "--param", default=(), help="Set extra params from CLI.", multiple=True
)
@click.option(
    "--debug-level", default=None, help="The debug level used during the build."
)
@click.option(
    "--use-experimental-kernels",
    is_flag=True,
    default=False,
    help="If enabled, then experimental kernels are used.",
)
@click.option(
    "-O",
    "--optimization-level",
    default=None,
    help="The optimization level used during the build.",
)
@click.option(
    "--target-accelerator",
    default=None,
    help="Specify the mojo target accelerator. Allowed values for this option:"
    + get_target_accelerator_helpstr(),
)
@click.option(
    "--disable-warnings",
    is_flag=True,
    default=False,
    help="Disable mojo build warnings.",
)
@click.option("--force", "-f", is_flag=True, default=False, help="Force.")
@click.option(
    "--cached",
    "-c",
    is_flag=True,
    default=False,
    help="Enable Kbench cache (WARNING: doesn't check for source changes).",
)
@click.option(
    "--clear-cache",
    "-cc",
    is_flag=True,
    default=False,
    help="Clear Kbench cache.",
)
@click.option(
    "--num-cpu",
    default=-1,
    help="Set the total number of cpu cores for building. Set to -1 for max number of cores (default=-1).",
)
@click.option(
    "--num-gpu",
    default=1,
    help="Set the total number of GPU devices for running, it can only be used with '--target-accelerator' (default=1).",
)
@click.option(
    "--dryrun",
    "-dryrun",
    is_flag=True,
    default=False,
    help="Do not execute the config, just show the parameters.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Verbose printing."
)
@click.option(
    "--shapes",
    default=(),
    help="Set of shapes passed as extra params.",
    multiple=True,
)
@click.option(
    "--build-opts",
    default="",
    help="Any build options (treated as str and directly passed to mojo compiler.)",
    multiple=False,
)
@click.option(
    "--profile",
    default=(),
    help="Set the profiler [ncu, ncu-single, rocm, rocprof-compute].",
    multiple=False,
)
@click.option(
    "--exec-prefix",
    default="",
    help="Any prefix options (treated as str and directly passed before binary.)",
    multiple=False,
)
@click.option(
    "--exec-suffix",
    default="",
    help="Any suffix options (treated as str and directly passed after binary.)",
    multiple=False,
)
@click.option(
    "--timeout-secs",
    default=None,
    help="Timeout seconds for executing each binary. (default=None)",
    multiple=False,
    type=click.INT,
)
@click.option(
    "--partition",
    default="0:1",
    help="Formatted as fraction 'm:n', divide the shapes "
    "into n partitions and limit the space to m'th partition "
    "(default='0:1' running everything). Note that it has no "
    "effect on parameter set and is only applied to shapes.",
    multiple=False,
    type=click.STRING,
)
@click.argument("files", nargs=-1, type=click.UNPROCESSED)
def cli(
    files,  # noqa: ANN001
    filter,  # noqa: ANN001
    output_path,  # noqa: ANN001
    output_dir,  # noqa: ANN001
    tune,  # noqa: ANN001
    build,  # noqa: ANN001
    param,  # noqa: ANN001
    debug_level,  # noqa: ANN001
    use_experimental_kernels,  # noqa: ANN001
    optimization_level,  # noqa: ANN001
    target_accelerator,  # noqa: ANN001
    disable_warnings,  # noqa: ANN001
    force,  # noqa: ANN001
    cached,  # noqa: ANN001
    clear_cache,  # noqa: ANN001
    num_cpu,  # noqa: ANN001
    num_gpu,  # noqa: ANN001
    dryrun,  # noqa: ANN001
    verbose,  # noqa: ANN001
    shapes,  # noqa: ANN001
    build_opts,  # noqa: ANN001
    profile,  # noqa: ANN001
    exec_prefix,  # noqa: ANN001
    exec_suffix,  # noqa: ANN001
    timeout_secs: int,
    partition: str,
) -> bool:
    configure_logging(verbose=verbose)

    if not verbose:
        sys.tracebacklimit = 1

    mode = KBENCH_MODE.RUN

    assert (build == False) or (tune == False)

    partition_idx, num_partitions = _validate_partition(partition)

    if build:
        mode = KBENCH_MODE.BUILD
    elif tune:
        mode = KBENCH_MODE.TUNE

    obj_cache = KbenchCache()
    # check kbench_cache and load it if exists:
    if clear_cache:
        obj_cache.clear()
    if cached:
        obj_cache.load()

    if not len(files) and not len(shapes):
        logging.info(
            "Nothing more to do without parameter or shape YAML provided!"
        )
        return True

    if not force:
        check_gpu_clock()

    # If `shapes` is not specified, pick an empty Spec and '-o output_path'.
    shape_list = list(Spec.load_yaml_list(shapes) if shapes else Spec())
    shape_path_list = (
        [Path(sh.hash(with_variables=True)) for sh in shape_list]
        if shapes
        else [Path(output_path)]
    )

    assert len(shape_path_list) == len(shape_list), (
        "Number of shapes doesn't equal number of paths."
    )

    if target_accelerator not in flatten(target_accelerator_values.values()) + [
        None
    ]:
        raise ValueError(
            f"--target-accelerator should be one of the following {get_target_accelerator_helpstr()}"
        )

    build_opts = build_opts.split(" ") if build_opts else []
    build_opts.extend(
        set_build_opts(
            debug_level,
            optimization_level,
            use_experimental_kernels,
            target_accelerator,
            disable_warnings,
        )
    )

    exec_suffix = exec_suffix.split(" ") if exec_suffix else []
    exec_prefix = exec_prefix.split(" ") if exec_prefix else []

    files = FileGlobArg(files) if files else []

    shapes_per_partition = math.ceil(len(shape_list) / num_partitions)
    shape_idx_lb = partition_idx * shapes_per_partition
    shape_idx_ub = min(shape_idx_lb + shapes_per_partition, len(shape_list))

    if num_gpu > 1 and not target_accelerator:
        raise ValueError(
            "Cannot use --num-gpu>1 without specifying --target-accelerator"
        )

    for i in range(shape_idx_lb, shape_idx_ub):
        run(
            yaml_path_list=files,
            obj_cache=obj_cache,
            shape=shape_list[i],
            output_path=shape_path_list[i],
            mode=mode,
            param_list=param,
            filter_list=filter,
            build_opts=build_opts,
            profile=profile,
            exec_prefix=exec_prefix,
            exec_suffix=exec_suffix,
            dryrun=dryrun,
            verbose=verbose,
            output_dir=output_dir,
            num_cpu=num_cpu,
            num_gpu=num_gpu,
            target_accelerator=target_accelerator,
            timeout_secs=timeout_secs,
        )
        if obj_cache.is_active:
            obj_cache.dump()
    logging.info(f"Number of shapes: {len(shape_list)}")
    return True


def main() -> None:
    try:
        cli()
    except Exception:
        CONSOLE.print_exception(suppress=[click, rich])


if __name__ == "__main__":
    if directory := os.environ.get("BUILD_WORKING_DIRECTORY"):
        os.chdir(directory)

    main()
