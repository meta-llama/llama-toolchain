# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from ogx.core.server.server import global_exception_handler, http_exception_handler
from ogx_api.common.errors import OpenAIErrorResponse, openai_error_type_for_status


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.get("/v1/registered")
    async def registered() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/v1/teapot")
    async def teapot() -> None:
        raise HTTPException(status_code=418, detail="I'm a teapot")

    @app.post("/v1alpha/interactions/boom")
    async def interactions_boom() -> None:
        raise HTTPException(status_code=404, detail="No such interaction")

    @app.get("/v1/kaboom")
    async def kaboom() -> None:
        raise ValueError("bad value")

    app.exception_handler(StarletteHTTPException)(http_exception_handler)
    app.exception_handler(Exception)(global_exception_handler)
    return TestClient(app, raise_server_exceptions=False)


def test_unregistered_path_returns_openai_error_shape(client: TestClient) -> None:
    """An unrouted path is what an OpenAI client hits against a partially configured stack."""
    response = client.get("/v1/conversations")

    assert response.status_code == 404
    assert response.json() == {"error": {"message": "Not Found", "type": "invalid_request_error"}}


def test_unsupported_method_returns_openai_error_shape(client: TestClient) -> None:
    response = client.post("/v1/registered")

    assert response.status_code == 405
    assert response.json() == {"error": {"message": "Method Not Allowed", "type": "invalid_request_error"}}
    # Starlette sets Allow on the exception it raises; the handler must not drop it.
    assert response.headers["allow"] == "GET"


def test_handler_http_exception_returns_openai_error_shape(client: TestClient) -> None:
    response = client.get("/v1/teapot")

    assert response.status_code == 418
    assert response.json() == {"error": {"message": "I'm a teapot", "type": "invalid_request_error"}}


def test_interactions_paths_keep_the_google_error_envelope(client: TestClient) -> None:
    response = client.post("/v1alpha/interactions/boom")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": 404, "message": "No such interaction"}}


def test_translated_exceptions_carry_an_error_type(client: TestClient) -> None:
    response = client.get("/v1/kaboom")

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    "status_code,expected",
    [
        (400, "invalid_request_error"),
        (401, "invalid_request_error"),
        (404, "invalid_request_error"),
        (429, "rate_limit_error"),
        (500, "server_error"),
        (503, "server_error"),
    ],
)
def test_error_type_for_status(status_code: int, expected: str) -> None:
    assert openai_error_type_for_status(status_code) == expected
    assert OpenAIErrorResponse.for_status(status_code, "boom").error.type == expected
