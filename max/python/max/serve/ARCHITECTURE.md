# Architecture

This document describes the architecture of the server. The server accepts HTTP
requests via endpoints and returns a model's response.

- `mocks`: Mock requests used in testing.
- `pipelines`: The logic glue between serving and the model worker.
- `router`: The server's HTTP routes.
- `scheduler`: The schedulers that batch requests in the model worker.
- `telemetry`
- `worker_interface`: The ZeroMQ queues between the API process and the model
  worker.
- `api_server.py`: The entrypoint to the server.

## Lifecycle of a Request

The following are the major points a request hits as it enters the server and
makes its way down to a model.

### 1. router/openai_routes.py

The request first enters the server via an HTTP endpoint.

#### OpenAIResponseGenerator

Handles the streaming of responses (via SSE) or completion of responses (via
JSON), depending on the endpoint requested. This wraps a
**TokenGeneratorPipeline**.

### 2. pipelines/llm.py

The response generator then requests either one chunk of tokens at a time or
all tokens from its pipeline.

#### TokenGeneratorPipeline

The base class for all LLM pipelines which provides an interface for retrieving
tokens either through `next_token_chunk` or `all_tokens`. It tokenizes the
request and sends it to the model worker through a **ModelWorkerProxy**.

### 3. worker_interface/

The pipeline then acts as a producer to the model worker's request queue, while
the model worker acts as a consumer, offloading to the underlying LLM model.
Requests, responses, and cancellations travel over ZeroMQ queues
(`worker_interface/_zmq_queue.py`). In the model worker, a scheduler from
`scheduler/` builds each batch.
