# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from fastapi.testclient import TestClient

from max.serve.pipelines.llm_pipeline_serving import app, models

client = TestClient(app)


def test_start_model():
    with client:
        response = client.post("/models/random/start")
        assert response.status_code < 300
        assert "random" in models(), response.content
        response = client.post("/models/random/stop")
        assert response.status_code < 300, response.content
        assert "random" not in models()


def test_stream_tokens():
    with client:
        response = client.post("/models/random/start")
        assert response.status_code < 300, response.content
        assert "random" in models()

        with client.stream("GET", "/models/random/generate?prompt=foo") as r:
            chunks = list(r.iter_raw())

        # I haven't quite figured this out yet, but they only come in 1 chunk.
        assert len(chunks) == 1
        # The following, using http.request works fine for streaming, but above doesn't
        #
        # import requests
        # response = requests.post('http://localhost:8000/models/random/start')
        # assert response.status_code < 300, response.content
        # response = requests.get('http://localhost:8000/models/random/generate?prompt=foo', stream=True, headers=None)
        # for i in response.iter_lines():
        #   print(i)
