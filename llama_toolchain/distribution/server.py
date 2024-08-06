# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import signal
from collections.abc import (
    AsyncGenerator as AsyncGeneratorABC,
    AsyncIterator as AsyncIteratorABC,
)
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    get_type_hints,
    List,
    Optional,
    Set,
)

import fire
import httpx
import yaml
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute

from pydantic import BaseModel, ValidationError
from termcolor import cprint

from .datatypes import Api, DistributionSpec, ProviderSpec, RemoteProviderSpec
from .distribution import api_endpoints
from .dynamic import instantiate_client, instantiate_provider

from .registry import resolve_distribution_spec

load_dotenv()


def is_async_iterator_type(typ):
    if hasattr(typ, "__origin__"):
        origin = typ.__origin__
        if isinstance(origin, type):
            return issubclass(
                origin,
                (AsyncIterator, AsyncGenerator, AsyncIteratorABC, AsyncGeneratorABC),
            )
        return False
    return isinstance(
        typ, (AsyncIterator, AsyncGenerator, AsyncIteratorABC, AsyncGeneratorABC)
    )


def create_sse_event(data: Any) -> str:
    if isinstance(data, BaseModel):
        data = data.json()
    else:
        data = json.dumps(data)

    return f"data: {data}\n\n"


async def global_exception_handler(request: Request, exc: Exception):
    http_exc = translate_exception(exc)

    return JSONResponse(
        status_code=http_exc.status_code, content={"error": {"detail": http_exc.detail}}
    )


def translate_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, ValidationError):
        return RequestValidationError(exc.raw_errors)

    # Add more custom exception translations here
    return HTTPException(status_code=500, detail="Internal server error")


async def passthrough(
    request: Request,
    downstream_url: str,
    downstream_headers: Optional[Dict[str, str]] = None,
):
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.update(downstream_headers or {})

    content = await request.body()

    async def iterating_response():
        def enc(x):
            return x.encode("latin-1")

        async with httpx.AsyncClient() as client:
            response_started = False
            try:
                async with client.stream(
                    method=request.method,
                    url=downstream_url,
                    headers=headers,
                    content=content,
                    params=request.query_params,
                ) as response:
                    yield enc(
                        f"HTTP/1.1 {response.status_code} {response.reason_phrase}\r\n"
                    )
                    for k, v in response.headers.items():
                        yield enc(f"{k}: {v}\r\n")
                    yield b"\r\n"

                    response_started = True

                    # using a small chunk size to allow for streaming SSE, this is not ideal
                    # for large responses but we are not in that regime for the most part
                    async for chunk in response.aiter_raw(chunk_size=64):
                        yield chunk
                    await response.aclose()
            except ReadTimeout:
                if not response_started:
                    yield enc(
                        "HTTP/1.1 504 Gateway Timeout\r\nContent-Type: text/plain\r\n\r\nDownstream server timed out"
                    )
                else:
                    yield enc("\r\n\r\nError: Downstream server timed out")
            except asyncio.CancelledError:
                print("Request cancelled")
                return
            except Exception as e:
                if not response_started:
                    yield enc(
                        f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\n\r\nError: {str(e)}"
                    )
                else:
                    yield enc(f"\r\n\r\nError: {e}")

    return StreamingResponse(iterating_response())


def handle_sigint(*args, **kwargs):
    print("SIGINT or CTRL-C detected. Exiting gracefully...")
    loop = asyncio.get_event_loop()
    for task in asyncio.all_tasks(loop):
        task.cancel()
    loop.stop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up")
    yield
    print("Shutting down")


def create_dynamic_passthrough(
    downstream_url: str, downstream_headers: Optional[Dict[str, str]] = None
):
    async def endpoint(request: Request):
        return await passthrough(request, downstream_url, downstream_headers)

    return endpoint


def create_dynamic_typed_route(func: Any):
    hints = get_type_hints(func)
    request_model = next(iter(hints.values()))
    response_model = hints["return"]

    # NOTE: I think it is better to just add a method within each Api
    # "Protocol" / adapter-impl to tell what sort of a response this request
    # is going to produce. /chat_completion can produce a streaming or
    # non-streaming response depending on if request.stream is True / False.
    is_streaming = is_async_iterator_type(response_model)

    if is_streaming:

        async def endpoint(request: request_model):
            async def sse_generator(event_gen):
                try:
                    async for item in event_gen:
                        yield create_sse_event(item)
                        await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    print("Generator cancelled")
                    await event_gen.aclose()
                except Exception as e:
                    print(e)
                    import traceback

                    traceback.print_exc()
                    yield create_sse_event(
                        {
                            "error": {
                                "message": str(translate_exception(e)),
                            },
                        }
                    )

            return StreamingResponse(
                sse_generator(func(request)), media_type="text/event-stream"
            )

    else:

        async def endpoint(request: request_model):
            try:
                return (
                    await func(request)
                    if asyncio.iscoroutinefunction(func)
                    else func(request)
                )
            except Exception as e:
                print(e)
                import traceback

                traceback.print_exc()
                raise translate_exception(e) from e

    return endpoint


def topological_sort(providers: List[ProviderSpec]) -> List[ProviderSpec]:

    by_id = {x.api: x for x in providers}

    def dfs(a: ProviderSpec, visited: Set[Api], stack: List[Api]):
        visited.add(a.api)

        if not isinstance(a, RemoteProviderSpec):
            for api in a.api_dependencies:
                if api not in visited:
                    dfs(by_id[api], visited, stack)

        stack.append(a.api)

    visited = set()
    stack = []

    for a in providers:
        if a.api not in visited:
            dfs(a, visited, stack)

    return [by_id[x] for x in stack]


def resolve_impls(dist: DistributionSpec, config: Dict[str, Any]) -> Dict[Api, Any]:
    provider_configs = config["providers"]
    provider_specs = topological_sort(dist.provider_specs.values())

    impls = {}
    for provider_spec in provider_specs:
        api = provider_spec.api
        if api.value not in provider_configs:
            raise ValueError(
                f"Could not find provider_spec config for {api}. Please add it to the config"
            )

        provider_config = provider_configs[api.value]
        if isinstance(provider_spec, RemoteProviderSpec):
            impls[api] = instantiate_client(
                provider_spec, provider_config.base_url.rstrip("/")
            )
        else:
            deps = {api: impls[api] for api in provider_spec.api_dependencies}
            impl = instantiate_provider(provider_spec, provider_config, deps)
            impls[api] = impl

    return impls


def main(yaml_config: str, port: int = 5000, disable_ipv6: bool = False):
    with open(yaml_config, "r") as fp:
        config = yaml.safe_load(fp)

    spec = config["spec"]
    dist = resolve_distribution_spec(spec)
    if dist is None:
        raise ValueError(f"Could not find distribution specification `{spec}`")

    app = FastAPI()

    all_endpoints = api_endpoints()
    impls = resolve_impls(dist, config)

    for provider_spec in dist.provider_specs.values():
        api = provider_spec.api
        endpoints = all_endpoints[api]
        impl = impls[api]

        if isinstance(provider_spec, RemoteProviderSpec):
            for endpoint in endpoints:
                url = impl.base_url + endpoint.route
                getattr(app, endpoint.method)(endpoint.route)(
                    create_dynamic_passthrough(url)
                )
        else:
            for endpoint in endpoints:
                if not hasattr(impl, endpoint.name):
                    # ideally this should be a typing violation already
                    raise ValueError(
                        f"Could not find method {endpoint.name} on {impl}!!"
                    )

                impl_method = getattr(impl, endpoint.name)
                getattr(app, endpoint.method)(endpoint.route, response_model=None)(
                    create_dynamic_typed_route(impl_method)
                )

    for route in app.routes:
        if isinstance(route, APIRoute):
            cprint(
                f"Serving {next(iter(route.methods))} {route.path}",
                "white",
                attrs=["bold"],
            )

    app.exception_handler(Exception)(global_exception_handler)
    signal.signal(signal.SIGINT, handle_sigint)

    import uvicorn

    # FYI this does not do hot-reloads
    listen_host = "::" if not disable_ipv6 else "0.0.0.0"
    print(f"Listening on {listen_host}:{port}")
    uvicorn.run(app, host=listen_host, port=port)


if __name__ == "__main__":
    fire.Fire(main)
