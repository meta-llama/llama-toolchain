# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json

import fire
import httpx
from termcolor import cprint

from .evals import *  # noqa: F403


class EvaluationClient(Evals):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def run_evals(
        self,
        model: str,
        task: str,
        dataset: Optional[str] = None,
        eval_task_config: Optional[EvaluateTaskConfig] = None,
    ) -> EvaluateResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/evals/run",
                json={
                    "model": model,
                    "task": task,
                    "dataset": dataset,
                    "eval_task_config": (
                        json.loads(eval_task_config.json())
                        if eval_task_config
                        else None
                    ),
                },
                headers={"Content-Type": "application/json"},
                timeout=3600,
            )
            response.raise_for_status()
            return EvaluateResponse(**response.json())


async def run_main(host: str, port: int):
    client = EvaluationClient(f"http://{host}:{port}")

    # CustomDataset
    response = await client.run_evals(
        model="Llama3.1-8B-Instruct",
        dataset="mmlu-simple-eval-en",
        task="mmlu",
        eval_task_config=EvaluateTaskConfig(
            n_samples=2,
        ),
    )
    cprint(f"evaluate response={response}", "green")

    # Eleuther Eval Task
    # response = await client.run_evals(
    #     model="Llama3.1-8B-Instruct",
    #     task="meta_mmlu_pro_instruct",
    #     # task="meta_ifeval",
    #     eval_task_config=EvaluateTaskConfig(
    #         n_samples=2,
    #     )
    # )
    # cprint(response.metrics["metrics_table"], "red")


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
