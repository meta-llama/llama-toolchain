# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import Mock

from ogx.core.datatypes import StackConfig
from ogx.core.server.server import ALWAYS_SERVED_APIS, apis_to_serve
from ogx_api import Api


def make_impls(*apis: Api) -> dict[Api, object]:
    return {api: Mock() for api in apis}


def test_absent_apis_list_serves_every_impl():
    config = StackConfig(distro_name="test", providers={})
    impls = make_impls(Api.inference, Api.responses)

    served = apis_to_serve(config, impls)

    assert {"inference", "responses"} <= served
    assert set(ALWAYS_SERVED_APIS) <= served


def test_empty_apis_list_serves_nothing_but_the_always_served_apis():
    config = StackConfig(distro_name="test", apis=[], providers={})
    impls = make_impls(Api.inference, Api.responses)

    assert apis_to_serve(config, impls) == set(ALWAYS_SERVED_APIS)


def test_explicit_apis_list_serves_only_what_it_names():
    config = StackConfig(distro_name="test", apis=["responses"], providers={})

    served = apis_to_serve(config, make_impls(Api.inference, Api.responses))

    assert "responses" in served
    assert "inference" not in served


def test_routing_table_api_follows_its_router_api():
    impls = make_impls(Api.inference, Api.models)

    with_inference = apis_to_serve(StackConfig(distro_name="test", apis=["inference"], providers={}), impls)
    without_inference = apis_to_serve(StackConfig(distro_name="test", apis=["files"], providers={}), impls)

    assert "models" in with_inference
    assert "models" not in without_inference
