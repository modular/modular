# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import logging
import math
import multiprocessing
import os
import queue
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import AsyncGenerator, Mapping

import uvloop
from max.pipelines import PipelinesFactory, TokenGenerator
from max.profiler import Tracer, traced
from max.serve.pipelines.llm import TokenGeneratorPipelineConfig
from max.serve.pipelines.scheduler import Scheduler
from max.serve.pipelines.scheduler_v2 import SchedulerConfig, SchedulerV2
from max.serve.scheduler.process_control import ProcessControl, ProcessMonitor
from max.serve.scheduler.queues import STOP_STREAM, EngineQueue
from max.serve.telemetry.metrics import METRICS, configure_metrics
from max.serve.telemetry.stopwatch import record_ms

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelWorkerConfig:
    worker_name: str = field(
        default_factory=lambda: str("MODEL_" + str(uuid.uuid4()))
    )
    timeout_secs: float = 20 * 60.0
    # Maximum time to wait for a heartbeat & remain "healthy"
    # This should be longer than ITL
    health_fail_s: float = 5.0


def _model_worker_process_fn(
    pc: ProcessControl,
    model_factory: PipelinesFactory,
    batch_config: TokenGeneratorPipelineConfig,
    worker_config: ModelWorkerConfig,
    queues: Mapping[str, Queue],
):
    try:
        uvloop.run(
            model_worker_run_v3(
                pc,
                model_factory,
                batch_config,
                worker_config,
                queues,
            )
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(
            "Encountered an error in _model_worker_process_fn %s",
            e,
            stack_info=True,
        )


@asynccontextmanager
async def start_model_worker(
    model_factory: PipelinesFactory,
    batch_config: TokenGeneratorPipelineConfig,
    config: ModelWorkerConfig = ModelWorkerConfig(),
) -> AsyncGenerator[EngineQueue, None]:
    """Starts a model worker and associated process.

    Args:
        factories (PipelinesFactory): Token generator factory functions.
        name (str, optional): Worker name. Defaults to "MODEL_<uuid>".

    Returns:
        AsyncIterator[Worker]: Iterator to model worker.

    Yields:
        Iterator[AsyncIterator[Worker]]: _description_
    """

    mp_context = multiprocessing.get_context("spawn")
    pc = ProcessControl(
        mp_context,
        "model-worker",
        health_fail_s=config.health_fail_s,
    )
    engine_queue: EngineQueue = EngineQueue(mp_context, pc=pc)
    queue_args = {
        "REQUEST": engine_queue.request_q,
        "RESPONSE": engine_queue.response_q,
        "CANCEL": engine_queue.cancel_q,
    }

    logger.info("Starting worker: %s", config.worker_name)
    worker = mp_context.Process(
        name=config.worker_name,
        target=_model_worker_process_fn,
        daemon=True,
        args=(
            pc,
            model_factory,
            batch_config,
            config,
            queue_args,
        ),
    )
    worker.start()
    monitor = ProcessMonitor(
        pc,
        worker,
        poll_s=10e-3,
        max_time_s=config.timeout_secs,
        unhealthy_poll_s=200e-3,
    )

    # before progressing, observe the worker process to be healthy or dead
    loop = asyncio.get_running_loop()
    ht = loop.create_task(monitor.until_healthy())
    dt = loop.create_task(monitor.until_dead())

    completed_tasks, pending_tasks = await asyncio.wait(
        [ht, dt],
        # Set a timeout longer than either task. This shouldn't be necessary, but being paranoid
        timeout=config.timeout_secs * 2,
        return_when=asyncio.FIRST_COMPLETED,
    )

    # cleanup tasks
    # observe the completed tasks
    for t in completed_tasks:
        await t
    # cancel the pending tasks
    for t in pending_tasks:
        t.cancel()

    # figure out if we are in a clean state
    # verify something completed
    if not ht.done() and not dt.done():
        # somehow neither task finished
        raise TimeoutError("Worker is neither dead nor healthy")

    # are we in a run-able state?
    if not worker.is_alive():
        # cannot continue if the worker is dead
        await monitor.shutdown()
        if pc.is_healthy():
            raise TimeoutError("Worker became healthy and died")
        else:
            raise TimeoutError("Worker died")

    # worker is alive!  it needs to be healthy too.

    if not pc.is_healthy():
        # cannot continue if the worker is not healthy
        await monitor.shutdown()
        raise TimeoutError("Worker did not become healthy")

    # worker is both alive and healthy!

    try:
        worker_task = loop.create_task(monitor.shutdown_if_unhealthy())
        yield engine_queue
    finally:
        worker_task.cancel()
        await monitor.shutdown()


# INTERNAL


@traced
async def model_worker_run_v2(
    pc: ProcessControl,
    model_factory: PipelinesFactory,
    batch_config: TokenGeneratorPipelineConfig,
    worker_config: ModelWorkerConfig,
    queues: Mapping[str, Queue],
):
    pc.set_started()
    configure_metrics()
    await METRICS.configure()
    try:
        tracer = Tracer()  # provides MojoTrace (NVTX spans
        pid = os.getpid()
        logger.info("Starting model worker on process %d!", pid)

        request_q = queues["REQUEST"]
        response_q = queues["RESPONSE"]
        cancel_q = queues["CANCEL"]

        logger.info("Worker Queues: %s", queues)

        # Initialize all token generators.
        with record_ms(METRICS.model_load_time), Tracer("model_factory"):
            model = model_factory()
            assert isinstance(model, TokenGenerator)

        scheduler = Scheduler(request_q, batch_config)
        logger.info("Token generators loaded!")

        logger.info("Started model worker!")

        i = 0
        # if cancelled allow some extra iterations to drain the queue
        # TODO: add a more structured drain-the-queue mechanism
        while i % 100 or not pc.is_canceled():
            pc.beat()
            i += 1

            prepared_batch = schedule(scheduler)

            if not prepared_batch.requests:
                await asyncio.sleep(0)
                continue

            batch_responses = next_token(model, prepared_batch, scheduler)

            handle_terminated_responses(
                prepared_batch.requests, batch_responses, model, scheduler
            )

            submit_responses(response_q, batch_responses)

            # Occasionally clear out contexts cancelled out API worker side.
            if i % 20 == 0 and not cancel_q.empty():
                handle_cancelled_requests(cancel_q, scheduler, model)

            await asyncio.sleep(0)

    except Exception as e:
        logger.exception("Failed worker process %d", pid)
        raise e

    finally:
        pc.set_completed()
        logger.info("Stopped model worker at process %d!", pid)


@traced
async def model_worker_run_v3(
    pc: ProcessControl,
    model_factory: PipelinesFactory,
    pipeline_config: TokenGeneratorPipelineConfig,
    worker_config: ModelWorkerConfig,
    queues: Mapping[str, Queue],
):
    configure_metrics()
    await METRICS.configure()

    pid = os.getpid()
    logger.info("Starting model worker on process %d!", pid)

    # Initialize token generator.
    with record_ms(METRICS.model_load_time), Tracer("model_factory"):
        token_generator_or_embedding = model_factory()
        assert isinstance(token_generator_or_embedding, TokenGenerator)
        pipeline: TokenGenerator = token_generator_or_embedding
    logger.info("Token generators loaded!")

    config = pipeline_config
    max_batch_size_tg = config.token_generation.size
    max_forward_steps_tg = config.token_generation.max_forward_steps
    target_tokens_per_batch_tg = config.token_generation.target_sum_seq_len
    if config.context_encoding:
        max_batch_size_ce = config.context_encoding.size
        max_forward_steps_ce = config.context_encoding.max_forward_steps
        target_tokens_per_batch_ce = config.context_encoding.target_sum_seq_len
        if math.isclose(config.context_encoding.timeout, 0.0):
            batch_timeout = None
        else:
            batch_timeout = config.context_encoding.timeout
    else:
        max_batch_size_ce = max_batch_size_tg
        max_forward_steps_ce = max_forward_steps_tg
        target_tokens_per_batch_ce = target_tokens_per_batch_tg
        batch_timeout = None

    scheduler_config = SchedulerConfig(
        max_batch_size_tg=max_batch_size_tg,
        max_forward_steps_tg=max_forward_steps_tg,
        target_tokens_per_batch_tg=target_tokens_per_batch_tg,
        max_batch_size_ce=max_batch_size_ce,
        max_forward_steps_ce=max_forward_steps_ce,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        batch_timeout=batch_timeout,
    )

    scheduler = SchedulerV2(
        process_control=pc,
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        queues=queues,
    )
    logger.info("Token generators loaded!")

    scheduler.pc.set_started()
    logger.info("Started model worker!")

    scheduler.run()

    scheduler.pc.set_completed()
    logger.info("Stopped model worker!")


@traced
def schedule(scheduler):
    return scheduler.schedule()


@traced
def next_token(model, prepared_batch, scheduler):
    output = model.next_token(
        prepared_batch.requests, num_steps=prepared_batch.num_steps
    )
    scheduler.step(prepared_batch.requests)
    return output


@traced
def submit_responses(response_q, batch_responses):
    response_q.put_nowait(batch_responses)


@traced
def handle_cancelled_requests(cancel_q, scheduler, model):
    try:
        for req_id in cancel_q.get_nowait():
            if not scheduler.contains(req_id):
                continue

            model.release(scheduler.get_request(req_id))
            scheduler.release(req_id)
    except queue.Empty:
        pass


@traced
def handle_terminated_responses(
    batch_executed, batch_responses, model, scheduler
):
    already_terminated = set()
    not_terminated = set(batch_executed.keys())

    for batch_response in batch_responses:
        terminated = not_terminated - batch_response.keys()
        for req_id in terminated:
            if req_id in already_terminated:
                continue

            model.release(batch_executed[req_id])

            batch_response[req_id] = STOP_STREAM
            already_terminated.add(req_id)
            not_terminated.remove(req_id)

    scheduler.release(already_terminated)
