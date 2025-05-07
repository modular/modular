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

import csv
import functools
import os
import pickle
import shutil
import string
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from time import time
from typing import Any

import click
import numpy as np
import pandas as pd
import rich
from model.utils.exceptions import CLIException, pretty_exception_handler
from modular.utils import logging, yaml
from modular.utils.subprocess import list2cmdline, run_shell_command
from modular.utils.yaml import YAML
from rich import print, traceback
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

CONSOLE = Console()
CURRENT_FILE = Path(__file__).resolve()
LINE = "\n" + 80 * "-"


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
        CONSOLE = Console(force_terminal=False, color_system=None)

    log_level = (
        logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    )
    logging.getLogger().setLevel(log_level)

    if verbose and pretty_output:
        traceback.install(suppress=[click, yaml, rich])
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


def _run_cmdline(cmd: list[str], dryrun: bool = False) -> ProcessOutput:
    """Execute a shell command with error handling."""
    try:
        if dryrun:
            print(list2cmdline(cmd))
            return ProcessOutput(None, None)

        output = run_shell_command(cmd, check=False, capture_output=True)
        return ProcessOutput(
            output.stdout.decode("utf-8"), output.stderr.decode("utf-8")
        )
    except Exception as exc:
        raise CLIException(
            f"Unable to run command {list2cmdline(cmd)}"
        ) from exc


@dataclass
class ParamSpace:
    name: str
    value: Any
    value_set: set[Any] = field(default_factory=set)
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
        self.value_set = sorted(set(flatten(self.value)))
        self.value = None
        self.length = len(self.value_set)


@dataclass
class ProcessOutput:
    stdout: str | None = None
    stderr: str | None = None
    path: Path | None = None

    def log(self):
        if self.stdout:
            logging.debug("output" + self.stdout + LINE)
        if self.stderr:
            logging.debug("error" + self.stderr + LINE)


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
        if self.path.exists():
            logging.debug(f"Removing kbench-cache: {self.path}")
            run_shell_command(["rm", str(self.path)])

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
        if obj_path in self.data.keys():
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

        defines = [item for sublist in defines for item in sublist]
        return defines

    @functools.cached_property
    def _get_vars(self) -> list[str]:
        vars = []
        for param in self.params:
            if param.name.startswith("$"):
                vars.append(param.define())

        vars = [item for sublist in vars for item in sublist]
        return vars

    def build(
        self,
        *,
        output_dir: Path,
        build_opts: list[str] = [],
        dryrun: bool = False,
        idx: int = -1,
    ) -> ProcessOutput:
        """Build the spec instance. Use set of compile-time
        parameters as path of the compiled binary and store
        the executable in 'output_dir'.
        """

        bin_name = self.hash(with_variables=False)
        bin_path = output_dir / Path(bin_name)

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
        if out.stderr:
            return ProcessOutput(out.stdout, out.stderr, None)
        return ProcessOutput(out.stdout, out.stderr, bin_path)

    def execute(
        self,
        binary_path: Path,
        output_file: Path,
        dryrun: bool = False,
        exec_prefix: list[str] = [],
        exec_suffix: list[str] = [],
    ) -> ProcessOutput:
        vars = self._get_vars
        cmd = []
        if exec_prefix:
            logging.debug(f"exec-prefix: {exec_prefix}")
            cmd.extend(exec_prefix)
        cmd.extend([binary_path, *vars, "-o", str(output_file)])
        if exec_suffix:
            cmd.extend(exec_suffix)
            logging.debug(f"exec-suffix: {exec_suffix}")
        out = _run_cmdline(cmd, dryrun)
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
        tokens = [self.file_stem]
        for param in self.params:
            name = param.name
            # just use compile-time parameters and ignore runtime variables.
            if name.startswith("$") and not with_variables:
                continue
            name = name.replace("$", "")
            tokens.append(f"{name}-{param.value}")
        return "_".join(tokens)


class GridSearchStrategy:
    instances: list[SpecInstance] = field(default_factory=list)

    def __init__(self, name, file, params):
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

    def __getitem__(self, i):
        return self.instances[i]

    def __len__(self):
        return len(self.instances)

    def extend(self, other):
        self.instances.extend(other.instances)


@dataclass(repr=True)
class Spec:
    name: str = ""
    file: Path = Path("")
    params: list[list[ParamSpace]] = field(default_factory=list)
    mesh_idx: int = 0
    mesh: list[SpecInstance] = field(default_factory=list)

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
        except Exception:
            raise ValueError(f"Could not load spec from {file}")

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
    def parse_params(param_list: list[str]):
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
        d = {}
        IFS = ":"
        for p in param_list:
            name = ""
            val = ""
            if IFS in p:
                name, val = p.split(IFS)

            if name not in d.keys():
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

    def extend_params(self, param_list: list[str]):
        # Expand with CLI params
        extra_params = self.parse_params(param_list)

        # For all params in each config either, update the exisiting `value_set``
        # with the new param value(s).
        for cfg in self.params:
            for k, v in extra_params.items():
                found = False
                for ps in cfg:
                    if ps.name == k:
                        ps.value_set.update(v)
                        found = True
                        break
                if not found:
                    cfg.append(ParamSpace(k, v))

        self.setup_mesh()

    def extend_shape_params(self, param_set: list[Param]):
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

    def dump_yaml(self, out_path: Path):
        assert self.mesh, "There are no instances to write to YAML!"
        obj = {
            "name": self.name,
            "file": self.file,
            "params": [s.to_obj() for s in self.mesh],
        }
        with open(out_path, "w") as f:
            YAML(typ="safe").dump(obj, f, sort=False)
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
        obj = YAML(typ="safe").load(yaml_str)

        if "name" not in obj.keys():
            logging.warning("Field [name] is not set in YAML")
        if "file" not in obj.keys():
            logging.warning("Field [file] is not set in YAML")

        params: list[list[ParamSpace]] = []
        if "params" in obj.keys():
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
        )

    def __len__(self):
        assert self.mesh
        return len(self.mesh)

    def __post_init__(self):
        # checking if the file source path is valid
        file_abs_path = Path(
            string.Template(str(self.file)).substitute(os.environ)
        ).absolute()
        assert file_abs_path.exists()
        self.file = file_abs_path

        # setup mesh
        if self.params:
            self.setup_mesh()
        else:
            # default values for empty mesh
            self.mesh = [SpecInstance("", Path("./"))]

    def setup_mesh(self):
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
        self.mesh = GridSearchStrategy(self.name, self.file, self.params)
        return len(self.mesh)

    def join(self, other: Spec):
        assert self.name == other.name
        assert self.file == other.file
        assert len(other.mesh) > 0

        self.mesh_idx = 0
        self.params.extend(other.params)
        self.mesh.extend(other.mesh)

    def filter(self, filter_list: list[str]):
        filters = {}
        for f in filter_list:
            if "=" in f:
                name, val = f.split("=")
            elif ":" in f:
                name, val = f.split(":")

            if name not in filters.keys():
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

        self.mesh.instances = filtered_insts[:]
        self.mesh_idx = 0

    def __iter__(self):
        self.iter_offset = 0
        return self

    def __next__(self) -> SpecInstance:
        assert self.mesh != None, (
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


def _get_tmp_path(file_path):
    base = os.path.basename(file_path).split(".")[0]
    tf = tempfile.NamedTemporaryFile(prefix=str(base) + "_").name + "/"
    return Path(tf)


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
        build_output: output message for build
        build_elapsed_time: elapsed time for build
        exec_output: output message for exec
    """

    idx: int
    spec_instance: SpecInstance
    output_dir: Path
    build_opts: list
    dryrun: bool = False
    output_path: Path = Path()

    build_output: ProcessOutput = ProcessOutput()
    build_elapsed_time: int = 0
    exec_output: ProcessOutput = ProcessOutput()


class Scheduler:
    """
    Kbench singleton scheduler class to coordinate building and running all items in spec.

    Args:
        num_cpu: number of cpu's (cores) used for building items
        build_items: list of spec items to build (BuildItem's)
        obj_cache: kbench obj-cache
        output_dir: parent output directory for all results
        num_specs: total number of spec items added to scheduler (to build+run)
    """

    num_cpu: int
    build_items: list[BuildItem]
    obj_cache: KbenchCache
    output_dir: Path
    num_specs: int

    def __init__(
        self,
        num_cpu: int,
        obj_cache: KbenchCache,
        spec_list: list[SpecInstance],
        output_dir: Path,
        build_opts: list[str],
        dryrun: bool,
        output_suffix: str = "output.csv",
        progress: Progress = Progress(),
    ):
        self.cpu_pool = Pool(num_cpu)
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
    def kbench_mkdir(output_dir):
        """Run the following command:
        `rm -rf {output_dir} && mkdir -p {output_dir}`
        """
        # "rm -rf {output_dir} && mkdir -p {output_dir}"
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=False)
        return output_dir

    def mk_output_dirs(self):
        """
        Make output directories for kbench results (one per spec-instance)
        """
        output_dir_list = [b.output_dir for b in self.build_items]
        res = self.cpu_pool.imap(self.kbench_mkdir, output_dir_list)
        [logging.debug(f"mkdir [{r}]") for r in res]
        logging.debug("Created directories for all instances in spec." + LINE)

    def schedule_unique_build_items(self) -> list[dict]:
        unique_build_items = {}
        unique_build_paths = {}
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
                if bin_name not in unique_build_items.keys():
                    unique_build_items[bin_name] = i
                    debug_msg += [f"Added to schedule (ref_idx=[{i}])"]
                else:
                    # Already in the unique_build_items list
                    idx = unique_build_items[bin_name]
                    debug_msg += [f"Currently in schedule (ref_idx=[{idx}])"]
            debug_msg = "\n".join(debug_msg)
            logging.debug(debug_msg + LINE)
        return [unique_build_items, unique_build_paths]

    @staticmethod
    def _pool_build_wrapper(bi: BuildItem) -> BuildItem:
        t_start_item = time()
        build_output = bi.spec_instance.build(
            output_dir=bi.output_dir,
            build_opts=bi.build_opts,
            dryrun=bi.dryrun,
            idx=bi.idx,
        )
        build_elapsed_time = int((time() - t_start_item) * 1e3)

        bi.build_output = build_output
        bi.build_elapsed_time = build_elapsed_time
        return bi

    def build_all(self):
        """
        Build all unique items scheduled by the scheduler.
        """

        unique_build_items, unique_build_paths = (
            self.schedule_unique_build_items()
        )
        unique_build_items = [
            self.build_items[i] for i in list(unique_build_items.values())
        ]

        logging.info(
            f"scheduled {len(unique_build_items)} unique build items out of {self.num_specs}"
            + LINE
        )

        if unique_build_items:
            obj_cache = self.obj_cache
            build_result = self.cpu_pool.map(
                self._pool_build_wrapper, unique_build_items
            )

            build_progress = self.progress.add_task(
                "build",
                total=len(unique_build_items),
            )

            for b in build_result:
                i = b.idx
                s = b.spec_instance

                build_output = b.build_output
                # update the data with build_output result
                self.build_items[i].build_output = build_output
                self.build_items[i].build_elapsed_time = b.build_elapsed_time

                bin_name = s.hash(with_variables=False)
                self.progress.update(
                    build_progress, description=f"building {str(bin_name)}"
                )

                # print build_output stdout and stderr using log function.
                build_output.log()

                # Try storing the executable in cache if:
                # - cache is active
                # - no error is reported in stderr
                # - build_output path is found
                if not build_output.stderr and build_output.path:
                    binary_path = build_output.path
                    obj_cache.store(bin_name, binary_path)
                    unique_build_paths[bin_name] = binary_path

                self.progress.update(build_progress, advance=1)
        return unique_build_paths

    def execute_all(self, unique_build_paths, exec_prefix, exec_suffix):
        """Execute all the items in the scheduler"""
        exec_progress = self.progress.add_task(
            "run",
            total=len(self.build_items),
        )

        for b in self.build_items:
            s = b.spec_instance
            bin_name = s.hash(with_variables=False)
            bin_path = unique_build_paths.get(bin_name, None)

            self.progress.update(exec_progress, description=f"run {str(s)}")

            if bin_path:
                exec_output = s.execute(
                    bin_path,
                    b.output_path,
                    dryrun=b.dryrun,
                    exec_prefix=exec_prefix,
                    exec_suffix=exec_suffix,
                )
                b.exec_output = b.exec_output
                exec_output.log()
            else:
                logging.error(f"Could not find binary [{bin_name}]")

            self.progress.update(exec_progress, advance=1)


def run(
    yaml_path_list,
    output_path=None,
    mode=KBENCH_MODE.RUN,
    param_list=None,
    shape: SpecInstance | None = None,
    filter_list=None,
    build_opts: list[str] = [],
    exec_prefix: list[str] = [],
    exec_suffix: list[str] = [],
    dryrun: bool = False,
    obj_cache: KbenchCache | None = None,
    verbose=False,
    output_dir=None,
    num_cpu=1,
):
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
        [logging.debug(f"[{i}]{s}") for i, s in enumerate(spec)]
        logging.debug(LINE)

    # Generate a tmp path for intermediate results.
    if not output_dir:
        output_dir = _get_tmp_path(spec.file)
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
    )

    # Set num_cpu to the maximum number of available CPUs
    if num_cpu == -1:
        num_cpu = len(os.sched_getaffinity(0))

    logging.info(f"num cpu's: {num_cpu}")
    # Kbench Singleton Scheduler
    scheduler = Scheduler(
        num_cpu=num_cpu,
        obj_cache=obj_cache,
        spec_list=list(spec),
        output_dir=output_dir,
        build_opts=build_opts,
        dryrun=dryrun,
        output_suffix="output.csv",
        progress=progress,
    )

    # Run the code over the mesh of param/values
    t_start_total = time()

    with progress:
        try:
            # Get the binary path for the unique list of build items
            # Build the binary if:
            # - could not find executable in the cache or cache is not active,
            # - could not find executable in the unique list of scheduled build items
            unique_build_paths = scheduler.build_all()
            if mode == KBENCH_MODE.RUN:
                scheduler.execute_all(
                    unique_build_paths,
                    exec_prefix=exec_prefix,
                    exec_suffix=exec_suffix,
                )
        except KeyboardInterrupt:
            obj_cache.dump()
            sys.exit(0)

    t_elapsed_total = time() - t_start_total
    output_lines = []
    output_dict = {}
    ########################################################
    # Elapsed time per spec
    build_df = pd.DataFrame(
        {
            "name": ["build" for b in scheduler.build_items],
            "spec": [f"{str(b.spec_instance)}" for b in scheduler.build_items],
        }
    )
    build_elapsed_time_list = [
        b.build_elapsed_time for b in scheduler.build_items
    ]
    build_df.insert(len(build_df.columns), "met (ms)", build_elapsed_time_list)
    build_df.insert(len(build_df.columns), "iters", 1)
    build_df["met (ms)"] = build_df["met (ms)"].fillna(0)

    build_df = build_df.loc[:, ["name", "met (ms)", "iters", "spec"]]

    output_dict["build_df"] = build_df
    if verbose:
        output_lines += [LINE]
        output_lines += ["Build time stats:"]
        output_lines += [build_df.to_string(index=False)]

    ########################################################
    # Retrieve, sort, and pick top choices
    valid_specs = []
    invalid_specs = []
    for b in scheduler.build_items:
        i = b.idx
        try:
            df = pd.read_csv(b.output_path, index_col=None, header=0)
            if not df.empty:
                df.insert(0, "mesh_idx", i)
                df.insert(len(df.columns), "spec", str(spec.mesh[i]))
                valid_specs.append(df)
            else:
                invalid_specs.append([i, b.build_output])

        except:
            invalid_specs.append([i, b.build_output])

    output_lines += [LINE]
    output_lines += [f"Running ['{spec.file}']"]

    if invalid_specs:
        output_lines += [LINE]
        output_lines += [
            f"Number of invalid specs: {len(invalid_specs)} (out of {len(spec)})"
        ]
        for idx, msg in invalid_specs:
            s = scheduler.build_items[idx].spec_instance
            output_lines += [LINE]
            output_lines += [f"mesh_idx: [{idx}][{s.to_obj()}]"]
            if msg.stdout:
                output_lines.append(msg.stdout)
            if msg.stderr:
                output_lines.append(msg.stderr)

    output_lines += [LINE]
    output_lines += [
        f"Number of valid executed specs: {len(valid_specs)} (out of {len(spec)})"
    ]

    if valid_specs:
        merged_df = pd.concat(valid_specs, axis=0, ignore_index=True)
        ########################################################
        # Get the name of column 2 (met (ms))
        output_dict["merged_df"] = merged_df
        met_col = merged_df.columns[2]
        if mode == KBENCH_MODE.TUNE:
            tune_df = merged_df.sort_values([met_col], ascending=True)

            output_dict["tune_df"] = tune_df
            output_lines += [tune_df.to_string(index=False)]
            # Index to top spec after sort
            top_spec_idx = tune_df.iloc[0].mesh_idx
            runtime = tune_df.iloc[0][met_col]

            output_lines += [LINE]
            output_lines += ["Spec with the best measured time:"]
            output_lines += [LINE]
            output_lines += [f"mesh_idx: {top_spec_idx} Runtime: {runtime}"]
            s = scheduler.build_items[top_spec_idx].spec_instance
            output_lines += [s.to_obj()]
            output_lines += [LINE]
        else:
            output_lines += [merged_df.to_string(index=False)]
            output_lines += [LINE]
        ########################################################

    output_lines += [f"Total elapsed running time: {t_elapsed_total:.3f} (s)"]
    output_str = "\n".join([str(x) for x in output_lines])
    print(output_str)

    if output_path:
        output_dict["name"] = spec.name
        output_dict["file"] = spec.file
        store_pickle(f"{output_path}.pkl", output_dict)

        # KBENCH_MODE.RUN overrides everything else and just dumps the running results.
        # THIS IS CRITICAL FOR CI automated kernel benchmarks workflow.
        if mode == KBENCH_MODE.RUN and valid_specs:
            merged_df.drop(columns=["mesh_idx"]).to_csv(
                output_path, index=False, quoting=csv.QUOTE_NONNUMERIC
            )
        elif mode == KBENCH_MODE.BUILD:
            build_df.to_csv(
                output_path, index=False, quoting=csv.QUOTE_NONNUMERIC
            )
        else:
            with open(output_path, "w") as f:
                f.write(output_str + "\n")

        logging.info(f"wrote results to [{output_path}]")
    logging.info(f"output-dir: [{output_dir}]")
    print(LINE + "\n\n")


@functools.cache
def get_nvidia_smi():
    return shutil.which("nvidia-smi")


def reset_gpu():
    nvidia_smi = get_nvidia_smi()
    if not nvidia_smi:
        return
    run_shell_command([nvidia_smi, "-r"])


def check_gpu_clock():
    nvidia_smi = get_nvidia_smi()
    if not nvidia_smi:
        return
    output = run_shell_command(
        [
            nvidia_smi,
            "--query-gpu",
            "persistence_mode",
            "--format",
            "csv",
        ],
        check=False,
        capture_output=True,
    )

    # We check for persistence here as a proxy to check if setup-gpu-benchmarking
    # has been run. This is not exact, but should cover most cases. Checking for
    # the clock frequency is more complicated since the frequencies changes per
    # GPU.
    if "Disabled" in output.stdout.decode("utf-8"):
        raise Exception(
            "the clock frequency for the GPU is not locked, please use"
            " `setup-gpu-benchmarking` to ensure that the frequencies and power"
            " of the GPU are locked to get consistent benchmarking behavior."
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

    def __len__(self):
        return len(self._files)


def set_build_opts(
    debug_level=None,
    optimization_level=None,
    use_experimental_kernels=None,
    target_accelerator=None,
    disable_warnings=None,
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


help_str = (
    "Grid-search all the params for a mojo benchmark and pick the top value"
)


@click.command(help=help_str, no_args_is_help=True)
@click.option(
    "--filter",
    "filter",
    help=(
        "Define a single filter (should match a valid paramter, can have"
        " multiple ones). The filters should of the format `--filter"
        " PARAM=VALUE`, that is, the subset of parameters that satisfy this"
        " condition will be included."
    ),
    multiple=True,
)
@click.option(
    "--output", "-o", "output_path", default=None, help="Path to output file."
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
    help="Tune by running and finding the best running time.",
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
    help="Specifiy the mojo target accelerator.",
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
    default=1,
    help="Set the total number of cpu cores for building. Set to -1 for max number of cores (default=1).",
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
    help="Set the profiler [ncu, ncu-single].",
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
@click.argument("files", nargs=-1, type=click.UNPROCESSED)
def cli(
    files: click.UNPROCESSED,
    filter,
    output_path,
    output_dir,
    tune,
    build,
    param,
    debug_level,
    use_experimental_kernels,
    optimization_level,
    target_accelerator,
    disable_warnings,
    force,
    cached,
    clear_cache,
    num_cpu,
    dryrun,
    verbose,
    shapes,
    build_opts,
    profile,
    exec_prefix,
    exec_suffix,
) -> bool:
    configure_logging(verbose=verbose)

    if not verbose:
        sys.tracebacklimit = 1

    mode = KBENCH_MODE.RUN

    assert (build == False) or (tune == False)
    if build:
        mode = KBENCH_MODE.BUILD
    elif tune:
        mode = KBENCH_MODE.TUNE

    if not force:
        check_gpu_clock()

    obj_cache = KbenchCache()

    # check kbench_cache and load it if exists:
    if clear_cache:
        obj_cache.clear()

    if cached:
        obj_cache.load()

    # If `shapes` is not specified, pick an empty Spec and '-o output_path'.
    shape_list = list(Spec.load_yaml_list(shapes)) if shapes else Spec()
    shape_path_list = (
        [sh.hash(with_variables=True) for sh in shape_list]
        if shapes
        else [output_path]
    )

    assert len(shape_path_list) == len(shape_list), (
        "Number of shapes doesn't equal number of paths."
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
    if profile in ["ncu", "ncu-single"]:
        exec_prefix.extend(["ncu"])
        if profile == "ncu-single":
            exec_suffix.extend(
                ["--bench-max-iters=0", "--bench-max-batch-size=1"]
            )

    files = FileGlobArg(files) if files else []
    for i, shape in enumerate(shape_list):
        run(
            yaml_path_list=files,
            output_path=shape_path_list[i],
            mode=mode,
            param_list=param,
            shape=shape,
            filter_list=filter,
            build_opts=build_opts,
            exec_prefix=exec_prefix,
            exec_suffix=exec_suffix,
            dryrun=dryrun,
            obj_cache=obj_cache,
            verbose=verbose,
            output_dir=output_dir,
            num_cpu=num_cpu,
        )
        if obj_cache.is_active:
            obj_cache.dump()
    logging.info(f"Number of shapes: {len(shape_list)}")
    return True


def main():
    try:
        cli()
    except Exception:
        CONSOLE.print_exception(suppress=[click, yaml, rich])


if __name__ == "__main__":
    main()
