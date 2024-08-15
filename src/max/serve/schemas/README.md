
# MAX Serve Schemas

Various schema files used to generate their corresponding Pydantic model classes.

## Regenerating

Run the following from the repo root directory.

### OpenAI

YAML [link](https://github.com/openai/openai-openapi/blob/master/openapi.yaml)

```shell
datamodel-codegen  --input SDK/lib/ServeAPI/python/max/serve/schemas/openai.yaml --input-file-type openapi --output-model-type pydantic_v2.BaseModel --output SDK/lib/ServeAPI/python/max/serve/schemas/openai.py  --enum-field-as-literal all
```
