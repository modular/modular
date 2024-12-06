# GRPC-KServe

This folder contains the gRPC-KServe implementation for max. The
 proto buffer definitions are in `ModelServing/proto/grpc_predict_v2.proto`.
The bazel targets`grpc_proto` generates the protobuffer and grpc
python wrappers, which the library `grpc_serve` depends on.
The entry point for gRPC serving is in
`SDK/lib/API/python/max/serve/grpc_server.py`
which depends on `grpc_serve` to launch an instance.

## Launching the server

The server can be launched via:

```shell
br //SDK/lib/API/python/max/serve:grpc_server -- --model=llama31
```

## Tests

There is a unit test in `SDK/test/API/serve/python/pytests/test_kserve_grpc.py`
which exercises all the end-points other than
the model inference. The test fixture use `pytest-grpc` which does
not support invoking async tests. However, the model inference methods
are async.

There are two ways to exercise the grpc end-points.

## 1. Via protobuf wrappers

This requires inclusion of the protobuf generated wrappers.
By using the wrappers, we can then launch as follows:

```python
import grpc
import grpc_kserve.grpc_predict_v2_pb2 as infer_pb2
import grpc_kserve.grpc_predict_v2_pb2_grpc as infer_grpc


def test_stream(idx, prompt_txt: str = "What is life?"):

    def create_request(text: str):
        response_contents = infer_pb2.InferTensorContents(
            int_contents=[ord(i) for i in text]
        )
        prompt_entry = infer_pb2.ModelInferRequest.InferInputTensor(
            name="prompt", contents=response_contents
        )
        return infer_pb2.ModelInferRequest(
            model_name="model_name",
            model_version="0",
            inputs=[prompt_entry],
        )

    with grpc.insecure_channel("localhost:9090") as channel:
        stub = infer_grpc.GRPCInferenceServiceStub(channel)
        request = create_request(prompt_txt)
        response = stub.ModelInfer(request)
        int_response = response.outputs[0].contents.int_contents
        print(f"{ascii_to_string(int_response)}", end="")

```

## 2. Via `grpcurl` utility

The [grpcurl](https://github.com/fullstorydev/grpcurl)
utility can be installed and passed json data which
will then be routed to server. For instance, the following
python snippet can be used to run a model inference call:

```python
def call_grpc_method(server_address, service, data):
    command = [
        "grpcurl",
        "-plaintext",
        "-d",
        data,
        server_address,
        f"{service}",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        print(f"Command error {result.returncode}: {result.stderr}")
    return result.stdout

def test_model_infer(model_name: str, prompt_txt: str):
    try:
        prompt_int = string_to_ascii(prompt_txt)
        data = (
            '{"model_name":"'
            + model_name
            + '", "parameters":{"max_tokens":{"string_param":"500"} } ,"inputs":[{"name":"prompt", "contents":{"int_contents":['
            + ",".join(str(i) for i in prompt_int)
            + "]}}] }"
        )
        response = call_grpc_method(
            "localhost:9090", "inference.GRPCInferenceService.ModelInfer", data
        )
        json_response = json.loads(response)
        response_int = json_response["outputs"][0]["contents"]["int_contents"]
        response_text = ascii_to_string(response_int)
        print(f"Response : {prompt_txt} ... {response_text}")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        print("Done")
```

## Streaming support

We have added a non-standard streaming version of the model inference call

```protobuf
rpc ModelInferStream(ModelInferRequest) returns (stream ModelInferResponse) {}
```

which returns a stream of responses instead of a single response.
Using the above example, we can replace `stub.ModelInfer` with
 `stub.ModelInferStream` to get streaming responses.

## Implementation notes

There are two routes to model inference via gRPC. The default route
is to use the serving API that ships with MAX and is also part of
the OpenAI end-point. This allows the gRPC offering to reuse the same
optimizations that are developed to accelerate the OpenAI offering.
The second route is via direct execution on a pipeline. This can be
enabled by passing `--bypass-serve` when launching the gRPC server.
The direct execution will not use the serving components and executes
the model inference calls directly on the pipeline interface. This is
mostly used by development to test model inference via gRPC.
