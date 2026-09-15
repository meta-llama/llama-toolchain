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
    impls = make_impls(Api.inference, Api.responses, Api.conversations)

    served = apis_to_serve(config, impls)

    assert {"inference", "responses", "conversations"} <= served
    assert set(ALWAYS_SERVED_APIS) <= served


def test_conversations_is_served_when_listed():
    config = StackConfig(distro_name="test", apis=["responses", "conversations"], providers={})

    assert "conversations" in apis_to_serve(config, make_impls(Api.responses, Api.conversations))


def test_conversations_is_not_served_when_omitted():
    """A gateway deployment that serves /v1/conversations itself must be able to turn it off."""
    config = StackConfig(distro_name="test", apis=["responses"], providers={})

    # The impl stays available in-process for providers that depend on it.
    served = apis_to_serve(config, make_impls(Api.responses, Api.conversations))

    assert "conversations" not in served
    assert "responses" in served


def test_administration_apis_are_served_even_when_omitted():
    config = StackConfig(distro_name="test", apis=["inference"], providers={})

    assert set(ALWAYS_SERVED_APIS) <= apis_to_serve(config, make_impls(Api.inference))


def test_routing_table_api_follows_its_router_api():
    impls = make_impls(Api.inference, Api.models)

    with_inference = apis_to_serve(StackConfig(distro_name="test", apis=["inference"], providers={}), impls)
    without_inference = apis_to_serve(StackConfig(distro_name="test", apis=["files"], providers={}), impls)

    assert "models" in with_inference
    assert "models" not in without_inference
