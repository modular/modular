# Architecture

This document describes the architecture of the server.  The server accepts HTTP
requests via endpoints and returns a model's response.

- `mocks`: Mock requests used in testing.
- `pipelines`: The logic glue between serving and the underlying queues.
- `router`: The server's HTTP routes.
- `scheduler`: The queues used to manage requests.
- `telemetry`
- `wheel`
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

The response generator then requests either one or all tokens from its pipeline.

#### TokenGeneratorPipeline

The base class for all LLM pipelines which provides an interface for retrieving
tokens either through `next_token` or `all_tokens`. This contains the queues
used for context encoding and token generation. Each pipeline can be configured
(via a **TokenGeneratorPipelineConfig**) as to how it utilizes its queues.

### 3. scheduler/queues.py

The pipeline then acts as a producer to the queues, while the workers act as
consumers offloading to the underlying LLM model.
