# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging
import time
import uuid
from concurrent import futures
from dataclasses import dataclass

import grpc

# mypy: disable-error-code="import-not-found"
import ModelServing.proto.grpc_predict_v2_pb2 as pb2
from typing import Callable
from grpc_reflection.v1alpha import reflection
from max.pipelines import PipelineConfig, TextTokenizer, PipelineTokenizer
from max.pipelines.interfaces import TokenGenerator, TokenGeneratorRequest
from max.serve.pipelines.llm import (
    TokenGeneratorPipeline,
    TokenGeneratorPipelineConfig,
)
from max.serve.pipelines.model_worker import start_model_worker
from ModelServing.proto.grpc_predict_v2_pb2_grpc import (
    GRPCInferenceServiceServicer,
    add_GRPCInferenceServiceServicer_to_server,
)

from cli import TextGenerationMetrics

# logging.root.setLevel(logging.INFO)


# TODO move this outside?
def get_batch_config(
    batch_size: int,  # Also KV-cache size.
    batch_timeout=0.0,
    max_forward_steps: int = 1,
) -> TokenGeneratorPipelineConfig:
    return TokenGeneratorPipelineConfig.continuous_heterogenous(
        tg_batch_size=batch_size,
        ce_batch_size=batch_size,
        ce_batch_timeout=batch_timeout,
        max_forward_steps=max_forward_steps,
    )


class MaxDirectInferenceService(GRPCInferenceServiceServicer):
    """Directly calls the pipeline method to execute a model."""

    def __init__(
        self,
        model_name: str,
        pipeline: TokenGenerator,
        tokenizer: PipelineTokenizer,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name
        self.pipeline = pipeline
        self.tokenizer = tokenizer

    def ServerLive(self, request: pb2.ServerLiveRequest, context):
        self.logger.debug(
            f"Server live called :: {request} ({type(request)}), context:"
            f" {context}"
        )
        is_live = True
        return pb2.ServerLiveResponse(live=is_live)

    async def ModelInfer(self, request: pb2.ModelInferRequest, context):
        self.logger.debug(f"Model infer called with {request}, {type(context)}")
        self.logger.debug(f"Request fields : {request.ListFields()}")
        model_name = request.model_name
        model_version = request.model_version
        if request.id:
            id = request.id
        else:
            id = str(uuid.uuid4())
        for name, param in request.parameters.items():
            print(f"PARAM:: name {name}, value : {param.string_param}")

        inputs = {}
        prompt_text = "what is the meaning of life?"
        for i in request.inputs:
            inputs[i.name] = i.contents.int_contents
            if i.name == "prompt":
                prompt_text = "".join(chr(c) for c in i.contents.int_contents)
        self.logger.info(f"Request data : {model_name}, {model_version}, {id}")

        try:
            num_tokens = 100
            tg_request = TokenGeneratorRequest(
                id=id,
                index=0,
                prompt=prompt_text,
                model_name=self.model_name,
            )
            text_context = await self.tokenizer.new_context(tg_request)
            batch = {id: text_context}
            text = ""
            for i in range(num_tokens):
                resp = self.pipeline.next_token(batch)
                if id in resp[0]:
                    text += await self.tokenizer.decode(
                        text_context, resp[0][id]
                    )
                else:
                    break
            self.logger.info(f"Response is {text}")
        except Exception as e:
            import traceback

            traceback.print_exception(e)
            self.logger.error(f"ERROR:: {e} :: ")
        finally:
            self.pipeline.release(text_context)

        response_contents = pb2.InferTensorContents(
            int_contents=[ord(i) for i in text]
        )
        response_entry = pb2.ModelInferResponse.InferOutputTensor(
            name="response", contents=response_contents
        )
        return pb2.ModelInferResponse(
            model_name=model_name,
            model_version=model_version,
            id=id,
            outputs=[response_entry],
        )

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

    def __init__(self, pipeline: TokenGeneratorPipeline):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pipeline = pipeline
        self.logger.info(f"Starting server handler")

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
        req_recv_time_ns = time.time_ns()
        self.logger.debug(f"Model infer called with {request}, {type(context)}")
        model_name = request.model_name
        model_version = request.model_version
        if request.id:
            id = request.id
        else:
            id = str(uuid.uuid4())
        max_tokens = 100
        for name, param in request.parameters.items():
            self.logger.debug(
                f"Request parameter:: name {name}, value : {param.string_param}"
            )
            if name == "max_tokens":
                max_tokens = int(param.string_param)

        inputs = {}
        prompt_text = "what is the meaning of life?"
        for i in request.inputs:
            inputs[i.name] = i.contents.int_contents
            if i.name == "prompt":
                prompt_text = "".join(chr(c) for c in i.contents.int_contents)
        self.logger.info(f"Request data : {model_name}, {model_version}, {id}")
        self.logger.debug(
            f"Max-tokens = {max_tokens}, Prompt is : {prompt_text}"
        )
        try:
            tg_request = await self.pipeline.create_request(
                id=id,
                prompt=prompt_text,
                model_name=self.pipeline.model_name,
                max_new_tokens=max_tokens,
                req_recv_time_ns=req_recv_time_ns,
            )
            self.logger.info(f"Request created : {tg_request}")
            text = ""
            async with self.pipeline:
                async for t in self.pipeline.next_token(tg_request):
                    text += t.decoded_token
                # text = await self.pipeline.all_tokens(tg_request)
            self.logger.info(f"Response is {text}")
        except Exception as e:
            import traceback

            self.logger.error(
                f"Error in processing {id} : {e} - {traceback.format_exc()}"
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(
                f"Error in processing {id} : {e} - {traceback.format_exc()}"
            )

        response_contents = pb2.InferTensorContents(
            int_contents=[ord(i) for i in text]
        )
        response_entry = pb2.ModelInferResponse.InferOutputTensor(
            name="response", contents=response_contents
        )
        return pb2.ModelInferResponse(
            model_name=model_name,
            model_version=model_version,
            id=id,
            outputs=[response_entry],
        )

    async def ServerReady(
        self, request: pb2.ServerReadyRequest, context
    ) -> pb2.ServerReadyResponse:
        return pb2.ServerReadyResponse(ready=self.pipeline is not None)

    async def ModelReady(
        self, request: pb2.ModelReadyRequest, context
    ) -> pb2.ModelReadyResponse:
        return pb2.ModelReadyResponse(
            ready=(request.name == self.pipeline.model_name)
        )

    async def ServerMetadata(
        self, request: pb2.ServerMetadataRequest, context
    ) -> pb2.ServerMetadataResponse:
        return pb2.ServerMetadataResponse(
            name="max-grpc-kserve", version="DEBUG"
        )

    async def ModelMetadata(
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

    tokenizer = TextTokenizer(pipeline_config)
    assert tokenizer.delegate
    batch_size = pipeline_config.max_cache_batch_size
    batch_config = get_batch_config(
        batch_size, max_forward_steps=pipeline_config.max_num_steps
    )
    pipeline = TokenGeneratorPipeline(batch_config, model_name, tokenizer)
    pipelines = {}
    pipelines[model_name] = pipeline
    dynamic_pipeline_config = TokenGeneratorPipelineConfig.dynamic_homogenous(
        batch_size=pipeline_config.max_cache_batch_size, batch_timeout=0
    )
    async with start_model_worker(
        factories={model_name: model_factory},
        configs={model_name: dynamic_pipeline_config},
    ):
        add_GRPCInferenceServiceServicer_to_server(
            MaxServeInferenceService(pipeline), server
        )
        server.add_insecure_port(f"[::]:{server_config.port}")
        await server.start()
        logging.info("Started server...")
        await server.wait_for_termination()


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

    with TextGenerationMetrics() as _:
        add_GRPCInferenceServiceServicer_to_server(
            MaxDirectInferenceService(model_name, pipeline, tokenizer), server
        )
        server.add_insecure_port(f"[::]:{server_config.port}")
        await server.start()
        logging.info("Started server...")
        await server.wait_for_termination()
