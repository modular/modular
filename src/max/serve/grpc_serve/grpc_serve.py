# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging
import queue
import traceback
import uuid
from concurrent import futures
from dataclasses import dataclass
from typing import Callable

import grpc
import max.serve.grpc_serve.grpc_predict_v2_pb2 as pb2
from grpc_reflection.v1alpha import reflection
from max.entrypoints.cli import TextGenerationMetrics
from max.pipelines import PipelineConfig, PipelineTokenizer, TextTokenizer
from max.pipelines.interfaces import TokenGenerator, TokenGeneratorRequest
from max.serve.grpc_serve.grpc_predict_v2_pb2_grpc import (
    GRPCInferenceServiceServicer,
    add_GRPCInferenceServiceServicer_to_server,
)
from max.serve.pipelines.llm import (
    TokenGeneratorPipeline,
    TokenGeneratorPipelineConfig,
)
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.telemetry.stopwatch import StopWatch

DEFAULT_MAX_TOKENS = 100


def tokengen_request_from_grpc_request(
    request: pb2.ModelInferRequest,
    index: int = 0,  # needed when bypassing serve
) -> TokenGeneratorRequest:
    """Creates a TokenGeneratorRequest object from a protobuf request instance.
    This method adds the request receive time, as well as converts the ascii
    input prompt to a text input prompt. The protobuf request contains an optional
    `max_tokens` to indicate the max output token length.
    """
    model_name = request.model_name
    max_tokens = DEFAULT_MAX_TOKENS
    sw = StopWatch()
    if request.id:
        id = request.id
    else:
        id = str(uuid.uuid4())
    for name, param in request.parameters.items():
        if name == "max_tokens":
            max_tokens = int(param.string_param)

    inputs = {}
    prompt_text = "what is the meaning of life?"
    for i in request.inputs:
        inputs[i.name] = i.contents.int_contents
        if i.name == "prompt":
            prompt_text = "".join(chr(c) for c in i.contents.int_contents)
    return TokenGeneratorRequest(
        id=id,
        index=index,
        prompt=prompt_text,
        model_name=model_name,
        max_new_tokens=max_tokens,
        timestamp_ns=sw.start_ns,
    )


def create_pb_response_from_text(
    tg_request: TokenGeneratorRequest, text: str
) -> pb2.ModelInferResponse:
    """Creates a protobuf model inference response for a given request object
    and a given text string. This can be a partial string used in streaming responses or
     a complete string in non-streaming responses."""
    response_contents = pb2.InferTensorContents(
        int_contents=[ord(i) for i in text]
    )
    response_entry = pb2.ModelInferResponse.InferOutputTensor(
        name="response", contents=response_contents
    )
    return pb2.ModelInferResponse(
        model_name=tg_request.model_name,
        # TODO rashid - support model version in the request? Plumbing seems overkill?
        model_version="0",
        id=tg_request.id,
        outputs=[response_entry],
    )


class MaxDirectInferenceService(GRPCInferenceServiceServicer):
    """Directly calls the pipeline method to execute a model."""

    def __init__(
        self,
        model_name: str,
        pipeline: TokenGenerator,
        tokenizer: PipelineTokenizer,
        max_batch_size: int,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        # This logger is too verbose to expose to end users. Disable propagation to the root logger by default.
        self.logger.propagate = False
        self.model_name = model_name
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.available_indexes = queue.Queue(max_batch_size)
        for i in range(max_batch_size):
            self.available_indexes.put_nowait(i)

    def ServerLive(self, request: pb2.ServerLiveRequest, context):
        self.logger.debug(
            f"Server live called :: {request} ({type(request)}), context:"
            f" {context}"
        )
        is_live = True
        return pb2.ServerLiveResponse(live=is_live)

    async def ModelInfer(self, request: pb2.ModelInferRequest, context):
        self.logger.debug(f"Model infer called with {request}")
        self.logger.debug(f"Request fields : {request.ListFields()}")
        try:
            index = self.available_indexes.get()
            tg_request = tokengen_request_from_grpc_request(request, index)
            self.logger.info(
                "Created token generator request instance : %s", tg_request
            )
            text_context = await self.tokenizer.new_context(tg_request)
            batch = {tg_request.id: text_context}
            text = ""
            max_tokens = (
                tg_request.max_new_tokens
                if tg_request.max_new_tokens
                else DEFAULT_MAX_TOKENS
            )
            for i in range(max_tokens):
                resp = self.pipeline.next_token(batch, num_steps=1)
                if tg_request.id in resp[0]:
                    text += await self.tokenizer.decode(
                        text_context, resp[0][tg_request.id].next_token
                    )
                else:
                    break
            self.logger.debug(f"Response is {text}")
        except Exception as e:
            self.logger.exception(
                "Error in model infer::%s", e, exc_info=True, stack_info=True
            )
        finally:
            self.logger.info("Completed request %s", tg_request)
            self.pipeline.release(text_context)
            self.available_indexes.put(index)
            return create_pb_response_from_text(tg_request, text)

    async def ModelInferStream(self, request: pb2.ModelInferRequest, context):
        self.logger.debug(f"Model infer called with {request}")
        self.logger.debug(f"Request fields : {request.ListFields()}")
        try:
            index = self.available_indexes.get()
            tg_request = tokengen_request_from_grpc_request(request, index)
            self.logger.info(
                "Created token generator request instance : %s", tg_request
            )
            num_tokens = (
                tg_request.max_new_tokens
                if tg_request.max_new_tokens
                else DEFAULT_MAX_TOKENS
            )
            text_context = await self.tokenizer.new_context(tg_request)
            batch = {tg_request.id: text_context}
            text = ""
            for i in range(num_tokens):
                resp = self.pipeline.next_token(batch, num_steps=1)
                if tg_request.id in resp[0]:
                    text += await self.tokenizer.decode(
                        text_context, resp[0][tg_request.id].next_token
                    )
                    yield create_pb_response_from_text(tg_request, text)
                else:
                    break
            self.logger.debug(f"Response is {text}")
        except Exception as e:
            self.logger.exception(
                "Error in model infer stream::%s",
                e,
                exc_info=True,
                stack_info=True,
            )
        finally:
            self.logger.info("Completed request :: %s", tg_request)
            self.pipeline.release(text_context)
            self.available_indexes.put(index)

    def ServerReady(
        self, request: pb2.ServerReadyRequest, context
    ) -> pb2.ServerReadyResponse:
        return pb2.ServerReadyResponse(ready=self.pipeline is not None)

    def ModelReady(
        self, request: pb2.ModelReadyRequest, context
    ) -> pb2.ModelReadyResponse:
        return pb2.ModelReadyResponse(ready=(request.name == self.model_name))

    def ServerMetadata(
        self, request: pb2.ServerMetadataRequest, context
    ) -> pb2.ServerMetadataResponse:
        return pb2.ServerMetadataResponse(
            name="max-grpc-kserve", version="DEBUG"
        )

    def ModelMetadata(
        self, request: pb2.ModelMetadataRequest, context
    ) -> pb2.ModelMetadataResponse:
        if request.name == self.model_name and request.version == "0":
            return pb2.ModelMetadataResponse(name=self.model_name)
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details(
            f"Model {request.name} version {request.version} not supported!"
        )
        raise NotImplementedError(
            f"Model {request.name} version {request.version} not supported!"
        )


class MaxServeInferenceService(GRPCInferenceServiceServicer):
    """Utilizes the max-serve infrastructure to run a pipeline."""

    def __init__(self, pipeline: TokenGeneratorPipeline, max_batch_size: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        # This logger is too verbose to expose to end users. Disable propagation to the root logger by default.
        self.logger.propagate = False
        self.pipeline = pipeline
        self.logger.info("Starting server handler")

    async def ServerLive(
        self, request: pb2.ServerLiveRequest, context
    ) -> pb2.ServerLiveResponse:
        self.logger.debug(
            f"Server live called :: {request} ({type(request)}), context:"
            f" {context}"
        )
        is_live = True
        return pb2.ServerLiveResponse(live=is_live)

    async def ModelInfer(
        self, request: pb2.ModelInferRequest, context
    ) -> pb2.ModelInferResponse:
        try:
            tg_request = tokengen_request_from_grpc_request(request)
            self.logger.info(f"Request created : {tg_request}")
            text = ""
            async for t in self.pipeline.next_token(tg_request):
                text += t.decoded_token
            return create_pb_response_from_text(tg_request, text)
        except Exception as e:
            error_msg = (
                f"Error in processing {id} : {e} - {traceback.format_exc()}"
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            self.logger.exception(
                "Error in model infer::%s", e, exc_info=True, stack_info=True
            )
        finally:
            self.logger.info(f"Request completed : {tg_request}")

    async def ModelInferStream(
        self, request: pb2.ModelInferRequest, context
    ) -> pb2.ModelInferResponse:
        try:
            tg_request = tokengen_request_from_grpc_request(request)
            self.logger.info(f"Request created : {tg_request}")
            async for t in self.pipeline.next_token(tg_request):
                yield create_pb_response_from_text(tg_request, t.decoded_token)
        except Exception as e:
            error_msg = (
                f"Error in processing {id} : {e} - {traceback.format_exc()}"
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            self.logger.exception(
                "Error in model infer::%s", e, exc_info=True, stack_info=True
            )
        finally:
            self.logger.info(f"Request completed : {tg_request}")

    def ServerReady(
        self, request: pb2.ServerReadyRequest, context
    ) -> pb2.ServerReadyResponse:
        return pb2.ServerReadyResponse(ready=self.pipeline is not None)

    def ModelReady(
        self, request: pb2.ModelReadyRequest, context
    ) -> pb2.ModelReadyResponse:
        return pb2.ModelReadyResponse(
            ready=(request.name == self.pipeline.model_name)
        )

    def ServerMetadata(
        self, request: pb2.ServerMetadataRequest, context
    ) -> pb2.ServerMetadataResponse:
        return pb2.ServerMetadataResponse(
            name="max-grpc-kserve", version="DEBUG"
        )

    def ModelMetadata(
        self, request: pb2.ModelMetadataRequest, context
    ) -> pb2.ModelMetadataResponse:
        if request.name == self.pipeline.model_name and request.version == "0":
            return pb2.ModelMetadataResponse(name=self.pipeline.model_name)
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details(
            f"Model {request.name} version {request.version} not supported!"
        )
        raise NotImplementedError(
            f"Model {request.name} version {request.version} not supported!"
        )


@dataclass
class GRPCConfig:
    port: int = 9090
    num_workers: int = 10
    max_batch_size: int = 8


async def grpc_serve(
    server_config: GRPCConfig,
    model_name: str,
    pipeline_config: PipelineConfig,
    model_factory: Callable[[], TokenGenerator],
):
    # TODO arekay - this would be very useful in making the server robust.
    # See https://grpc-interceptor.readthedocs.io/en/latest/
    # interceptors = [ExceptionToStatusInterceptor()]
    # logger = logging.getLogger("ServerStartup")
    # logger.setLevel(logging.INFO)
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(
            max_workers=10
        ),  # , interceptors=interceptors
    )
    logging.info(
        "Services available ::"
        f" {[k for k in pb2.DESCRIPTOR.services_by_name.keys()]}"
    )
    SERVICE_NAMES = (
        pb2.DESCRIPTOR.services_by_name["GRPCInferenceService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    tokenizer = TextTokenizer(
        pipeline_config.model_path,
        pipeline_config.max_length,
        pipeline_config.max_new_tokens,
        pipeline_config.trust_remote_code,
    )
    assert tokenizer.delegate

    cont_batching = TokenGeneratorPipelineConfig.continuous_heterogenous(
        tg_batch_size=server_config.max_batch_size,
        ce_batch_size=1,
    )
    server_settings = Settings()
    try:
        async with (
            start_model_worker(
                model_factory=model_factory,
                batch_config=cont_batching,
                settings=server_settings,
            ) as worker_queue,
            TokenGeneratorPipeline(
                model_name, tokenizer, worker_queue
            ) as pipeline,
        ):
            if not worker_queue.is_worker_healthy():
                logging.error("Worker process not healthy")
                exit(-1)
            pipelines = {}
            pipelines[model_name] = pipeline
            add_GRPCInferenceServiceServicer_to_server(
                MaxServeInferenceService(
                    pipeline, server_config.max_batch_size
                ),
                server,
            )
            server.add_insecure_port(f"[::]:{server_config.port}")
            await server.start()
            logging.info("Started server (via serve API)...")
            await server.wait_for_termination()
    except Exception as ex:
        logging.exception("Exception in grpc_serve %s", ex)
    finally:
        await server.stop(None)
        logging.info("Shutting down!")


async def grpc_serve_direct(
    server_config: GRPCConfig,
    model_name: str,
    tokenizer: PipelineTokenizer,
    pipeline: TokenGenerator,
):
    # logger = logging.getLogger("ServerStartup")
    # TODO arekay - this would be very useful in making the server robust.
    # See https://grpc-interceptor.readthedocs.io/en/latest/
    # interceptors = [ExceptionToStatusInterceptor()]
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(
            max_workers=server_config.num_workers
        )  # , interceptors=interceptors
    )
    logging.info(
        "Services available ::"
        f" {[k for k in pb2.DESCRIPTOR.services_by_name.keys()]}"
    )
    SERVICE_NAMES = (
        pb2.DESCRIPTOR.services_by_name["GRPCInferenceService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    try:
        with TextGenerationMetrics() as _:
            add_GRPCInferenceServiceServicer_to_server(
                MaxDirectInferenceService(
                    model_name,
                    pipeline,
                    tokenizer,
                    server_config.max_batch_size,
                ),
                server,
            )
            server.add_insecure_port(f"[::]:{server_config.port}")
            await server.start()
            logging.info("Started server (direct)...")
            await server.wait_for_termination()
    except Exception as ex:
        logging.error("Exception encountered ", ex)
    finally:
        logging.info("Terminating....")
        await server.stop(None)
