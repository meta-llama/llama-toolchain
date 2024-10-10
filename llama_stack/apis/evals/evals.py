# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol

from llama_models.schema_utils import webmethod

from pydantic import BaseModel

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.dataset import *  # noqa: F403


class EvaluationJob(BaseModel):
    job_uuid: str


class EvaluationJobLogStream(BaseModel):
    job_uuid: str


class EvaluateTaskConfig(BaseModel):
    # num examples to evaluate, evaluate all if None
    n_samples: Optional[int] = None
    # model evaluation params
    sampling_params: SamplingParams = SamplingParams()


@json_schema_type
class EvaluateResponse(BaseModel):
    """Scores for evaluation."""

    metrics: Dict[str, str]


@json_schema_type
class EvaluationJobStatusResponse(BaseModel):
    job_uuid: str


@json_schema_type
class EvaluationJobArtifactsResponse(BaseModel):
    """Artifacts of a evaluation job."""

    job_uuid: str


@json_schema_type
class EvaluationJobCreateResponse(BaseModel):
    """Response to create a evaluation job."""

    job_uuid: str


class Evals(Protocol):
    @webmethod(route="/evals/run")
    async def run_evals(
        self,
        model: str,
        task: str,
        dataset: Optional[str] = None,
        eval_task_config: Optional[EvaluateTaskConfig] = None,
    ) -> EvaluateResponse: ...

    # @webmethod(route="/evals/jobs")
    # def get_evaluation_jobs(self) -> List[EvaluationJob]: ...

    # @webmethod(route="/evals/job/create")
    # async def create_evaluation_job(
    #     self, model: str, dataset: str, task: str
    # ) -> EvaluationJob: ...

    # @webmethod(route="/evals/job/status")
    # def get_evaluation_job_status(
    #     self, job_uuid: str
    # ) -> EvaluationJobStatusResponse: ...

    # # sends SSE stream of logs
    # @webmethod(route="/evals/job/logs")
    # def get_evaluation_job_logstream(self, job_uuid: str) -> EvaluationJobLogStream: ...

    # @webmethod(route="/evals/job/cancel")
    # def cancel_evaluation_job(self, job_uuid: str) -> None: ...

    # @webmethod(route="/evals/job/artifacts")
    # def get_evaluation_job_artifacts(
    #     self, job_uuid: str
    # ) -> EvaluationJobArtifactsResponse: ...
