
# MAX Serve Schemas

Various schema files used to generate their corresponding Pydantic model classes.

## Regenerating

Supported protocols:

- `openai`: [OpenAPI YAML](https://github.com/openai/openai-openapi/blob/master/openapi.yaml)
- `kserve`: [OpenAPI YAML](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/rest_predict_v2.yaml)

Run the following from the repo root directory.

```shell
datamodel-codegen \
--input-file-type openapi \
--enum-field-as-literal all \
--output-model-type pydantic_v2.BaseModel \
--input SDK/lib/ServeAPI/python/max/serve/schemas/<protocol>.yaml \
--output SDK/lib/ServeAPI/python/max/serve/schemas/<protocol>.py
```
