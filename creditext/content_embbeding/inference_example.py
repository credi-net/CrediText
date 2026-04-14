from openai import OpenAI
from propella import (
    create_messages,
    AnnotationResponse,
    get_annotation_response_schema,
)

document = "Hi, its me Max."

client = OpenAI(base_url="http://localhost:6060/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="ellamind/propella-1-4b",
    messages=create_messages(document),
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "AnnotationResponse",
            "schema": get_annotation_response_schema(flatten=True, compact_whitespace=True),
            "strict": True,
        }
    },
)
response_content = response.choices[0].message.content
result = AnnotationResponse.model_validate_json(response_content)
print(result.model_dump_json(indent=4))
