# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import logging
import multiprocessing
import os
import queue
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from multiprocessing import Queue
from multiprocessing.synchronize import Event
from typing import AsyncGenerator, Mapping

import uvloop
from max.pipelines import PipelinesFactory, TokenGenerator
from max.profiler import Tracer, traced
from max.serve.pipelines.llm import TokenGeneratorPipelineConfig
from max.serve.pipelines.scheduler import Scheduler
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
    enable_health_check: bool = False
    health_check_sleep_sec: float = 5.0


def _model_worker_process_fn(
    model_factory: PipelinesFactory,
    pipeline_config: TokenGeneratorPipelineConfig,
    worker_config: ModelWorkerConfig,
    queues: Mapping[str, Queue],
    events: Mapping[str, Event],
):
    try:
        uvloop.run(
            model_worker_run_v2(
                model_factory, pipeline_config, worker_config, queues, events
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
    pipeline_config: TokenGeneratorPipelineConfig,
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
    engine_queue: EngineQueue = EngineQueue(context=mp_context)
    queue_args = {
        "REQUEST": engine_queue.request_q,
        "RESPONSE": engine_queue.response_q,
        "CANCEL": engine_queue.cancel_q,
        "HEALTH": engine_queue.health_q,
    }
    started_event = mp_context.Event()  # has the worker started
    stopped_event = mp_context.Event()  # has the worker stopped
    shutdown_event = mp_context.Event()  # should the worker be shutdown
    event_args = {
        "STARTED": started_event,
        "STOPPED": stopped_event,
        "SHUTDOWN": shutdown_event,
    }

    logger.info("Starting worker: %s", config.worker_name)
    worker = mp_context.Process(
        name=config.worker_name,
        target=_model_worker_process_fn,
        daemon=True,
        args=(
            model_factory,
            pipeline_config,
            config,
            queue_args,
            event_args,
        ),
    )
    worker.start()
    engine_queue.pid = worker.pid if worker.pid else -1
    engine_queue.process = worker

    async def worker_health():
        try:
            last_health_check = time.time()
            curr_time = last_health_check
            while True:
                await asyncio.sleep(config.health_check_sleep_sec)
                msgs = []
                # Drain all messages from the queue
                while True:
                    try:
                        msg = engine_queue.health_q.get_nowait()
                        msgs.append(msg)
                    except queue.Empty:
                        break

                curr_time = time.time()
                if msgs:
                    logger.debug("HEALTH_MESSAGE :: %s", msgs)
                    last_health_check = curr_time
                if (
                    curr_time
                    > last_health_check + config.health_check_sleep_sec
                ):
                    raise TimeoutError(
                        "Health check failed with timeout %s",
                        config.health_check_sleep_sec,
                    )
                for m in msgs:
                    if m != "OK":
                        raise RuntimeError("Model worker corrupted")
        except TimeoutError:
            logger.exception("Model worker timeout")
        finally:
            logger.info("Exiting health check task")
            shutdown_event.set()
            if worker.is_alive():
                worker.kill()

    async def worker_started():
        while not started_event.is_set():
            await asyncio.sleep(0.01)

    async def worker_completed():
        while worker.is_alive():
            await asyncio.sleep(0.01)

    # Wait for one of the following tasks to complete.
    # 1. The worker signals started()
    # 2. The worker task completes - likely a failure
    loop = asyncio.get_running_loop()
    completed_tasks, pending_tasks = await asyncio.wait(
        [
            loop.create_task(worker_started()),
            loop.create_task(worker_completed()),
        ],
        timeout=config.timeout_secs,
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Handle timeout
    if not completed_tasks:
        shutdown_event.set()
        for p in pending_tasks:
            p.cancel()
        raise TimeoutError(
            f"Startup timed out for model worker {config.worker_name}."
        )

    # Observe completed task result.
    # This will either be the startup or the completed event
    for t in completed_tasks:
        await t

    try:
        if config.enable_health_check:
            worker_task = loop.create_task(worker_health())
        yield engine_queue
    finally:
        if config.enable_health_check:
            worker_task.cancel()
        shutdown_event.set()
        if worker.is_alive():
            worker.kill()


# INTERNAL


@traced
async def model_worker_run_v2(
    model_factory: PipelinesFactory,
    pipeline_config: TokenGeneratorPipelineConfig,
    worker_config: ModelWorkerConfig,
    queues: Mapping[str, Queue],
    events: Mapping[str, Event],
):
    configure_metrics()
    await METRICS.configure()
    try:
        tracer = Tracer()  # provides MojoTrace (NVTX spans
        pid = os.getpid()
        logger.info("Starting model worker on process %d!", pid)

        # Multiprocessing resources.
        started = events["STARTED"]
        stopped = events["STOPPED"]
        shutdown = events["SHUTDOWN"]

        request_q = queues["REQUEST"]
        response_q = queues["RESPONSE"]
        cancel_q = queues["CANCEL"]
        health_q = queues["HEALTH"]

        logger.info("Worker Queues: %s, Events: %s", queues, events)

        # Initialize all token generators.
        with record_ms(METRICS.model_load_time), Tracer("model_factory"):
            model = model_factory()
            assert isinstance(model, TokenGenerator)

        scheduler = Scheduler(request_q, pipeline_config)
        logger.info("Token generators loaded!")

        started.set()
        logger.info("Started model worker!")

        i = 0
        while i % 100 or not shutdown.is_set():
            i += 1
            # TODO arekay - configure frequency of health updates
            if worker_config.enable_health_check and (i % 1_000_000 == 0):
                try:
                    health_q.put_nowait("OK")  # TODO make this better
                except queue.Full as e:
                    logger.exception("health check queue is full %s", e)

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
        stopped.set()
        logger.info("Stopped model worker at process %d!", pid)


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
