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

import gc
import logging
import math
import os
import sys
from pathlib import Path
from time import time

import click
import pandas as pd
import rich
from kbench_model import KBENCH_MODE, KbenchCache, Scheduler, Spec, SpecInstance
from rich import traceback
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from utils import (
    LINE,
    _get_core_count,
    _get_visible_device_prefix,
    _percentage,
    check_gpu_clock,
    check_valid_target_accelerator,
    get_target_accelerator_helpstr,
    pretty_exception_handler,
)

##### Utilities and configurations #####

CONSOLE = Console(width=80)
CURRENT_FILE = Path(__file__).resolve()

pd.set_option("display.float_format", str)


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


def log_and_raise_error(message: str, param_hint: str | None = None) -> None:
    """Log an error and raise a Click exception.

    Args:
        message: The error message to log and display
        param_hint: Optional parameter hint for BadParameter exception
    """
    logging.error(message)
    if param_hint:
        raise click.BadParameter(message, param_hint=param_hint)
    else:
        raise click.UsageError(message)


def run(
    yaml_path_list,  # noqa: ANN001
    obj_cache: KbenchCache,
    shape: SpecInstance,
    output_path: Path = Path(),
    mode=KBENCH_MODE.BUILD_AND_RUN,  # noqa: ANN001
    param_list=None,  # noqa: ANN001
    filter_list=None,  # noqa: ANN001
    build_opts: list[str] = [],  # noqa: B006
    profile: str = "",
    exec_prefix: list[str] = [],  # noqa: B006
    exec_suffix: list[str] = [],  # noqa: B006
    dryrun: bool = False,
    verbose: bool = False,
    output_dir: Path | None = None,
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

    # Set output_dir='./kbench-output' if it is not specified.
    if not output_dir:
        output_dir = Path("./kbench-output")

    # Set output_path (for storing results) relative to output_dir
    output_path = output_dir / output_path
    os.makedirs(output_path.parent, exist_ok=True)
    # strip output_path suffix
    if output_path.suffix in [".csv", ".pkl", ".txt"]:
        output_path = output_path.with_suffix("")

    if shape:
        spec.extend_shape_params(shape.params)
        # Each shape should have its own temporary directory.
        output_dir = output_dir / Path(shape.hash(with_variables=True))

    logging.info(f"output-dir: [{output_dir}]")

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
        run_only=(mode == KBENCH_MODE.RUN),
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
            obj_cache.dump()

            t_build_total = time() - t_start_total

            t_benchmark_start = time()
            if mode in [KBENCH_MODE.RUN, KBENCH_MODE.BUILD_AND_RUN]:
                scheduler.setup_execution_pool()
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


def _validate_partition(partition: str) -> list[int]:
    assert ":" in partition
    partition_idx, num_partitions = [int(x) for x in partition.split(":")]
    assert num_partitions > 0
    assert 0 <= partition_idx < num_partitions, (
        "Condition: 0 <= partition_idx < num_partitions"
    )
    return [partition_idx, num_partitions]


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


@click.command(
    help="Benchmarking toolkit for Mojo kernels", no_args_is_help=True
)
@click.option(
    "--filter",
    "filter",
    help=(
        "Define a single filter (should match a valid parameter, can have"
        " multiple ones). The filters should of the format '--filter"
        " PARAM=VALUE', that is, the subset of parameters that satisfy this"
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
    default="kbench-output",
    help="Path to output directory for all results (default='./kbench-output')",
)
@click.option(
    "--build",
    "build",
    is_flag=True,
    default=False,
    help="Just build the binary and report the build time.",
)
@click.option(
    "--run-only",
    "run_only",
    is_flag=True,
    default=False,
    help="Only run, do not build. Cache must exist, -c is implied",
)
@click.option(
    "--param",
    default=(),
    help="Set extra params in the format of 'PARAM:VALUE'. Example: '--param use_vendor_blas:True'",
    multiple=True,
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
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="This option will be deprecated soon. See --skip-clock-check instead.",
)
@click.option(
    "--skip-clock-check",
    is_flag=True,
    default=False,
    help="Run even if accelerator clocks are not set to maximum.",
)
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
    "--mpirun-np",
    default=1,
    help="Set the total number of GPU devices for running with mpirun, it cannot be combined with '--num-gpus' (default=1)."
    "Make sure to call 'Bench.check_mpirun()' in mojo benchmark.",
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
    build,  # noqa: ANN001
    run_only,  # noqa: ANN001
    param,  # noqa: ANN001
    debug_level,  # noqa: ANN001
    use_experimental_kernels,  # noqa: ANN001
    optimization_level,  # noqa: ANN001
    target_accelerator,  # noqa: ANN001
    disable_warnings,  # noqa: ANN001
    force,  # noqa: ANN001
    skip_clock_check,  # noqa: ANN001
    cached,  # noqa: ANN001
    clear_cache,  # noqa: ANN001
    num_cpu,  # noqa: ANN001
    num_gpu,  # noqa: ANN001
    mpirun_np: int,
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

    if force:
        logging.warning(
            "'--force' option is deprecated and will be removed soon. Please use '--skip-clock-check' instead."
        )
        skip_clock_check = True

    mode = KBENCH_MODE.BUILD_AND_RUN

    if run_only:
        mode = KBENCH_MODE.RUN

    if run_only and clear_cache:
        log_and_raise_error(
            "Cannot clear cache when in run-only mode. Need cache to run.",
            param_hint="'--clear-cache'",
        )
    if run_only and build_opts:
        log_and_raise_error("Cannot provide build options when run-only mode")

    partition_idx, num_partitions = _validate_partition(partition)

    if build:
        mode = KBENCH_MODE.BUILD

    obj_cache = KbenchCache()
    # check kbench_cache and load it if exists:
    if clear_cache and run_only:
        log_and_raise_error("Trying to clear cache when run_only")
    elif clear_cache:
        obj_cache.clear()

    if cached or (mode == KBENCH_MODE.RUN):
        obj_cache.load()

    if len(obj_cache.data) == 0 and mode == KBENCH_MODE.RUN:
        log_and_raise_error(
            "Run Only requires an active cache object but the object is empty",
            param_hint="'--run-only'",
        )

    if not len(files) and not len(shapes):
        logging.info(
            "Nothing more to do without parameter or shape YAML provided!"
        )
        return True

    # Resolve YAML file paths from the input user globs
    yaml_files = []
    for f in files:
        yaml_files.append(Path(f).resolve())

    if len(yaml_files) == 0:
        log_and_raise_error(
            "No valid YAML files found from the input globs.",
            param_hint="'FILES'",
        )

    if not skip_clock_check:
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

    if target_accelerator and not check_valid_target_accelerator(
        target_accelerator
    ):
        log_and_raise_error(
            f"Invalid target accelerator '{target_accelerator}'. "
            f"Should be one of the following {get_target_accelerator_helpstr()}",
            param_hint="'--target-accelerator'",
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

    if num_gpu > 1 and not target_accelerator:
        raise ValueError(
            "Cannot use --num-gpu>1 without specifying --target-accelerator"
        )
    if mpirun_np > 1 and num_gpu > 1:
        raise ValueError(
            "Cannot use --num-gpu>1 and --mpirun-np>1 at the same time!"
        )

    exec_suffix = exec_suffix.split(" ") if exec_suffix else []
    exec_prefix = exec_prefix.split(" ") if exec_prefix else []
    if mpirun_np > 1:
        exec_prefix.extend(["mpirun", "-np", str(mpirun_np)])

    shapes_per_partition = math.ceil(len(shape_list) / num_partitions)
    shape_idx_lb = partition_idx * shapes_per_partition
    shape_idx_ub = min(shape_idx_lb + shapes_per_partition, len(shape_list))

    for i in range(shape_idx_lb, shape_idx_ub):
        run(
            yaml_path_list=yaml_files,
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
