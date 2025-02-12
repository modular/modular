# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from max.serve.router import openai_routes
from max.serve.schemas.openai import CreateCompletionRequest  # type: ignore


def test_prompts():
    completion_req = CreateCompletionRequest(
        model="whatev",
        prompt="Why is the sky blue?",
    )
    prompts = openai_routes.openai_get_prompts_from_completion_request(
        completion_req
    )
    assert len(prompts) == 1

    completion_req = CreateCompletionRequest(
        model="whatev",
        prompt=["Why is the sky blue?", "what time is it?"],
    )
    prompts = openai_routes.openai_get_prompts_from_completion_request(
        completion_req
    )
    assert len(prompts) == 2

    completion_req = CreateCompletionRequest(
        model="whatev",
        prompt=[[1, 2, 3]],
    )
    prompts = openai_routes.openai_get_prompts_from_completion_request(
        completion_req
    )
    assert len(prompts) == 1

    completion_req = CreateCompletionRequest(
        model="whatev",
        prompt=[1, 2, 3],
    )
    prompts = openai_routes.openai_get_prompts_from_completion_request(
        completion_req
    )
    assert len(prompts) == 1

    # prompt item
    completion_req = CreateCompletionRequest(
        model="whatev",
        prompt=[[1, 2, 3]],
    )
    prompts = openai_routes.openai_get_prompts_from_completion_request(
        completion_req
    )
    assert len(prompts) == 1

    completion_req = CreateCompletionRequest(
        model="whatev",
        prompt=[[1, 2, 3], [4, 5, 6]],
    )
    prompts = openai_routes.openai_get_prompts_from_completion_request(
        completion_req
    )
    assert len(prompts) == 2
