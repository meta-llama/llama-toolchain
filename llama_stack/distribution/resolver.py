# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import importlib
import inspect

from typing import Any, Dict, List, Set

from llama_stack.providers.datatypes import *  # noqa: F403
from llama_stack.distribution.datatypes import *  # noqa: F403

from llama_stack.apis.agents import Agents
from llama_stack.apis.evals import Evals
from llama_stack.apis.inference import Inference
from llama_stack.apis.inspect import Inspect
from llama_stack.apis.memory import Memory
from llama_stack.apis.memory_banks import MemoryBanks
from llama_stack.apis.models import Models
from llama_stack.apis.safety import Safety
from llama_stack.apis.shields import Shields
from llama_stack.apis.telemetry import Telemetry
from llama_stack.distribution.distribution import (
    builtin_automatically_routed_apis,
    get_provider_registry,
)
from llama_stack.distribution.utils.dynamic import instantiate_class_type


def api_protocol_map() -> Dict[Api, Any]:
    return {
        Api.agents: Agents,
        Api.inference: Inference,
        Api.inspect: Inspect,
        Api.memory: Memory,
        Api.memory_banks: MemoryBanks,
        Api.models: Models,
        Api.safety: Safety,
        Api.shields: Shields,
        Api.telemetry: Telemetry,
        Api.evals: Evals,
    }


def additional_protocols_map() -> Dict[Api, Any]:
    return {
        Api.inference: ModelsProtocolPrivate,
        Api.memory: MemoryBanksProtocolPrivate,
        Api.safety: ShieldsProtocolPrivate,
    }


# TODO: make all this naming far less atrocious. Provider. ProviderSpec. ProviderWithSpec. WTF!
class ProviderWithSpec(Provider):
    spec: ProviderSpec


# TODO: this code is not very straightforward to follow and needs one more round of refactoring
async def resolve_impls_with_routing(run_config: StackRunConfig) -> Dict[Api, Any]:
    """
    Does two things:
    - flatmaps, sorts and resolves the providers in dependency order
    - for each API, produces either a (local, passthrough or router) implementation
    """
    all_api_providers = get_provider_registry()

    routing_table_apis = set(
        x.routing_table_api for x in builtin_automatically_routed_apis()
    )
    router_apis = set(x.router_api for x in builtin_automatically_routed_apis())

    providers_with_specs = {}

    for api_str, providers in run_config.providers.items():
        api = Api(api_str)
        if api in routing_table_apis:
            raise ValueError(
                f"Provider for `{api_str}` is automatically provided and cannot be overridden"
            )

        specs = {}
        for provider in providers:
            if provider.provider_type not in all_api_providers[api]:
                raise ValueError(
                    f"Provider `{provider.provider_type}` is not available for API `{api}`"
                )

            p = all_api_providers[api][provider.provider_type]
            p.deps__ = [a.value for a in p.api_dependencies]
            spec = ProviderWithSpec(
                spec=p,
                **(provider.dict()),
            )
            specs[provider.provider_id] = spec

        key = api_str if api not in router_apis else f"inner-{api_str}"
        providers_with_specs[key] = specs

    apis_to_serve = run_config.apis or set(
        list(providers_with_specs.keys())
        + [x.value for x in routing_table_apis]
        + [x.value for x in router_apis]
    )

    for info in builtin_automatically_routed_apis():
        if info.router_api.value not in apis_to_serve:
            continue

        available_providers = providers_with_specs[f"inner-{info.router_api.value}"]

        providers_with_specs[info.routing_table_api.value] = {
            "__builtin__": ProviderWithSpec(
                provider_id="__routing_table__",
                provider_type="__routing_table__",
                config={},
                spec=RoutingTableProviderSpec(
                    api=info.routing_table_api,
                    router_api=info.router_api,
                    module="llama_stack.distribution.routers",
                    api_dependencies=[],
                    deps__=([f"inner-{info.router_api.value}"]),
                ),
            )
        }

        providers_with_specs[info.router_api.value] = {
            "__builtin__": ProviderWithSpec(
                provider_id="__autorouted__",
                provider_type="__autorouted__",
                config={},
                spec=AutoRoutedProviderSpec(
                    api=info.router_api,
                    module="llama_stack.distribution.routers",
                    routing_table_api=info.routing_table_api,
                    api_dependencies=[info.routing_table_api],
                    deps__=([info.routing_table_api.value]),
                ),
            )
        }

    sorted_providers = topological_sort(
        {k: v.values() for k, v in providers_with_specs.items()}
    )
    apis = [x[1].spec.api for x in sorted_providers]
    sorted_providers.append(
        (
            "inspect",
            ProviderWithSpec(
                provider_id="__builtin__",
                provider_type="__builtin__",
                config={
                    "run_config": run_config.dict(),
                },
                spec=InlineProviderSpec(
                    api=Api.inspect,
                    provider_type="__builtin__",
                    config_class="llama_stack.distribution.inspect.DistributionInspectConfig",
                    module="llama_stack.distribution.inspect",
                    api_dependencies=apis,
                    deps__=([x.value for x in apis]),
                ),
            ),
        )
    )

    print(f"Resolved {len(sorted_providers)} providers")
    for api_str, provider in sorted_providers:
        print(f" {api_str} => {provider.provider_id}")
    print("")

    impls = {}
    inner_impls_by_provider_id = {f"inner-{x.value}": {} for x in router_apis}
    for api_str, provider in sorted_providers:
        deps = {a: impls[a] for a in provider.spec.api_dependencies}

        inner_impls = {}
        if isinstance(provider.spec, RoutingTableProviderSpec):
            inner_impls = inner_impls_by_provider_id[
                f"inner-{provider.spec.router_api.value}"
            ]

        impl = await instantiate_provider(
            provider,
            deps,
            inner_impls,
        )
        # TODO: ugh slightly redesign this shady looking code
        if "inner-" in api_str:
            inner_impls_by_provider_id[api_str][provider.provider_id] = impl
        else:
            api = Api(api_str)
            impls[api] = impl

    return impls


def topological_sort(
    providers_with_specs: Dict[str, List[ProviderWithSpec]],
) -> List[ProviderWithSpec]:
    def dfs(kv, visited: Set[str], stack: List[str]):
        api_str, providers = kv
        visited.add(api_str)

        deps = []
        for provider in providers:
            for dep in provider.spec.deps__:
                deps.append(dep)

        for dep in deps:
            if dep not in visited:
                dfs((dep, providers_with_specs[dep]), visited, stack)

        stack.append(api_str)

    visited = set()
    stack = []

    for api_str, providers in providers_with_specs.items():
        if api_str not in visited:
            dfs((api_str, providers), visited, stack)

    flattened = []
    for api_str in stack:
        for provider in providers_with_specs[api_str]:
            flattened.append((api_str, provider))
    return flattened


# returns a class implementing the protocol corresponding to the Api
async def instantiate_provider(
    provider: ProviderWithSpec,
    deps: Dict[str, Any],
    inner_impls: Dict[str, Any],
):
    protocols = api_protocol_map()
    additional_protocols = additional_protocols_map()

    provider_spec = provider.spec
    module = importlib.import_module(provider_spec.module)

    args = []
    if isinstance(provider_spec, RemoteProviderSpec):
        if provider_spec.adapter:
            method = "get_adapter_impl"
        else:
            method = "get_client_impl"

        config_type = instantiate_class_type(provider_spec.config_class)
        config = config_type(**provider.config)
        args = [config, deps]
    elif isinstance(provider_spec, AutoRoutedProviderSpec):
        method = "get_auto_router_impl"

        config = None
        args = [provider_spec.api, deps[provider_spec.routing_table_api], deps]
    elif isinstance(provider_spec, RoutingTableProviderSpec):
        method = "get_routing_table_impl"

        config = None
        args = [provider_spec.api, inner_impls, deps]
    else:
        method = "get_provider_impl"

        config_type = instantiate_class_type(provider_spec.config_class)
        config = config_type(**provider.config)
        args = [config, deps]

    fn = getattr(module, method)
    impl = await fn(*args)
    impl.__provider_id__ = provider.provider_id
    impl.__provider_spec__ = provider_spec
    impl.__provider_config__ = config

    check_protocol_compliance(impl, protocols[provider_spec.api])
    if (
        not isinstance(provider_spec, AutoRoutedProviderSpec)
        and provider_spec.api in additional_protocols
    ):
        additional_api = additional_protocols[provider_spec.api]
        check_protocol_compliance(impl, additional_api)

    return impl


def check_protocol_compliance(obj: Any, protocol: Any) -> None:
    missing_methods = []

    mro = type(obj).__mro__
    for name, value in inspect.getmembers(protocol):
        if inspect.isfunction(value) and hasattr(value, "__webmethod__"):
            if not hasattr(obj, name):
                missing_methods.append((name, "missing"))
            elif not callable(getattr(obj, name)):
                missing_methods.append((name, "not_callable"))
            else:
                # Check if the method signatures are compatible
                obj_method = getattr(obj, name)
                proto_sig = inspect.signature(value)
                obj_sig = inspect.signature(obj_method)

                proto_params = set(proto_sig.parameters)
                proto_params.discard("self")
                obj_params = set(obj_sig.parameters)
                obj_params.discard("self")
                if not (proto_params <= obj_params):
                    print(
                        f"Method {name} incompatible proto: {proto_params} vs. obj: {obj_params}"
                    )
                    missing_methods.append((name, "signature_mismatch"))
                else:
                    # Check if the method is actually implemented in the class
                    method_owner = next(
                        (cls for cls in mro if name in cls.__dict__), None
                    )
                    if (
                        method_owner is None
                        or method_owner.__name__ == protocol.__name__
                    ):
                        missing_methods.append((name, "not_actually_implemented"))

    if missing_methods:
        raise ValueError(
            f"Provider `{obj.__provider_id__} ({obj.__provider_spec__.api})` does not implement the following methods:\n{missing_methods}"
        )
