# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import dataclasses
from typing import Dict, Optional, Union

import datasets

from lm_eval.tasks.ifeval import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(
    inp,
    response,
):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    response,
):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results(doc, results):
    new_kwargs = []
    for item in doc["kwargs"]:
        if item["nth_paragraph"]:
            item["nth_paragraph"] = int(item["nth_paragraph"])
        new_kwargs.append(item)
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=new_kwargs,
    )
    response = results[0]

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _get_question(example: dict) -> dict:
        # get the question from the ifeval dataset
        example["input_question"] = (
            eval(
                example["input_question"]
                .replace("null", "None")
                .replace("true", "True")
                .replace("false", "False")
            )["dialog"][0]["body"]
            .replace("Is it True that the first song", "Is it true that the first song")
            .replace("Is the following True", "Is the following true")
        )
        example["input_final_prompts"] = example["input_final_prompts"][0]
        return example

    original_dataset_name = "wis-k/instruction-following-eval"
    ifeval_data = datasets.load_dataset(original_dataset_name, split="train")
    ifeval_df = ifeval_data.to_pandas()
    ifeval_df = ifeval_df.rename(columns={"prompt": "input_question"})

    meta_dataset = dataset.map(_get_question)
    meta_df = meta_dataset.to_pandas()

    # join the two datasets on the input_question column
    joined = meta_df.join(ifeval_df.set_index("input_question"), on="input_question")
    joined = joined.rename(columns={"input_final_prompts": "prompt"})
    joined = joined.rename(columns={"is_correct": "previous_is_correct"})
    joined = datasets.Dataset.from_pandas(joined)
    joined = joined.select_columns(
        [
            "input_question",
            "prompt",
            "previous_is_correct",
            "instruction_id_list",
            "kwargs",
            "output_prediction_text",
            "key",
        ]
    )
    joined.rename_column("output_prediction_text", "previous_output_prediction_text")
    return joined
