"""
Router behavior the router-code differential fuzz run found changed once the Rust core became the
only core: count settings below zero, an allowed_fails_policy write that is no policy, a model
response object as a deployment's mock_response, and endpoints read or patched on the Router class.
"""

from typing import Final
from unittest import mock

import pytest

import litellm
from litellm import Router
from litellm.types.utils import ModelResponse

MESSAGES: Final = [{"role": "user", "content": "hi"}]


def _deployment(name: str, reply: object) -> dict[str, object]:
    return {"model_name": name, "litellm_params": {"model": "openai/gpt-4o", "api_key": "k", "mock_response": reply}}


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["num_retries", "max_fallbacks", "allowed_fails"])
async def test_a_negative_count_setting_builds_and_serves(setting: str) -> None:
    router: Final = Router(model_list=[_deployment("g", "ok")], **{setting: -1})

    response: Final = await router.acompletion(model="g", messages=MESSAGES)

    assert response.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_a_negative_max_fallbacks_takes_no_fallback() -> None:
    router: Final = Router(
        model_list=[_deployment("g", Exception("boom")), _deployment("fb", "fb")],
        fallbacks=[{"g": ["fb"]}],
        max_fallbacks=-1,
        num_retries=0,
    )

    with pytest.raises(litellm.InternalServerError):
        await router.acompletion(model="g", messages=MESSAGES)


@pytest.mark.asyncio
async def test_a_negative_num_retries_still_falls_back() -> None:
    router: Final = Router(
        model_list=[_deployment("g", Exception("boom")), _deployment("fb", "fb")],
        fallbacks=[{"g": ["fb"]}],
        num_retries=-1,
    )

    response: Final = await router.acompletion(model="g", messages=MESSAGES)

    assert response.choices[0].message.content == "fb"


@pytest.mark.asyncio
async def test_an_allowed_fails_policy_written_as_a_list_reads_as_no_policy() -> None:
    router: Final = Router(model_list=[_deployment("g", "ok")])

    with pytest.warns(DeprecationWarning, match="allowed_fails_policy"):
        router.allowed_fails_policy = []  # pyright: ignore[reportAttributeAccessIssue]  # the old attribute write
    response: Final = await router.acompletion(model="g", messages=MESSAGES)

    assert response.choices[0].message.content == "ok"
    assert router.settings().allowed_fails_policy == []


def test_a_model_response_mock_reads_back_from_the_model_list_as_a_dict() -> None:
    reply: Final = ModelResponse(id="chatcmpl-fixed", choices=[{"message": {"role": "assistant", "content": "fb"}}])
    router: Final = Router(model_list=[_deployment("g", reply)])

    params: Final = router.get_model_list()[0]["litellm_params"]

    assert isinstance(params["mock_response"], dict)
    assert params["mock_response"]["id"] == "chatcmpl-fixed"


def test_an_endpoint_the_core_serves_by_name_is_readable_on_the_class() -> None:
    endpoint: Final = Router.aretrieve_batch

    assert callable(endpoint)
    assert endpoint is Router.aretrieve_batch
    assert endpoint.__name__ == "aretrieve_batch"


@pytest.mark.asyncio
async def test_a_class_level_patch_of_an_endpoint_reaches_the_instance_and_unwinds() -> None:
    router: Final = Router(model_list=[_deployment("g", "ok")])

    with mock.patch.object(Router, "asearch", new=mock.AsyncMock(return_value="patched")) as patched:
        assert await router.asearch(query="q", search_tool_name="t") == "patched"
    patched.assert_awaited_once()

    assert router.asearch is not patched
