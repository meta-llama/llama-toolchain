# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from abc import ABC, abstractmethod
from urllib.parse import parse_qs, urlparse

import pandas
from datasets import Dataset, load_dataset


class BaseDataset(ABC):
    def __init__(self, name: str):
        self.dataset = None
        self.dataset_id = name
        self.type = self.__class__.__name__

    def __iter__(self):
        return iter(self.dataset)

    @abstractmethod
    def load(self):
        pass


class CustomDataset(BaseDataset):
    def __init__(self, name, url):
        super().__init__(name)
        self.url = url

    def load(self):
        if self.dataset:
            return
        # TODO: better support w/ data url
        if self.url.endswith(".csv"):
            df = pandas.read_csv(self.url)
        elif self.url.endswith(".xlsx"):
            df = pandas.read_excel(self.url)

        self.dataset = Dataset.from_pandas(df)


class HFDataset(BaseDataset):
    def __init__(self, name, url):
        super().__init__(name)
        self.url = url

    def load(self):
        if self.dataset:
            return

        parsed = urlparse(self.url)

        if parsed.scheme != "hf":
            raise ValueError(f"Unknown HF dataset: {self.url}")

        query = parse_qs(parsed.query)
        query = {k: v[0] for k, v in query.items()}
        path = parsed.netloc
        self.dataset = load_dataset(path, **query)
