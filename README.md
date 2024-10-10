# Llama Stack

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama_stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![Discord](https://img.shields.io/discord/1257833999603335178)](https://discord.gg/llama-stack)

This repository contains the Llama Stack API specifications as well as API Providers and Llama Stack Distributions.

The Llama Stack defines and standardizes the building blocks needed to bring generative AI applications to market. These blocks span the entire development lifecycle: from model training and fine-tuning, through product evaluation, to building and running AI agents in production. Beyond definition, we are building providers for the Llama Stack APIs. These were developing open-source versions and partnering with providers, ensuring developers can assemble AI solutions using consistent, interlocking pieces across platforms. The ultimate goal is to accelerate innovation in the AI space.

The Stack APIs are rapidly improving, but still very much work in progress and we invite feedback as well as direct contributions.


## APIs

The Llama Stack consists of the following set of APIs:

- Inference
- Safety
- Memory
- Agentic System
- Evaluation
- Post Training
- Synthetic Data Generation
- Reward Scoring

Each of the APIs themselves is a collection of REST endpoints.


## API Providers

A Provider is what makes the API real -- they provide the actual implementation backing the API.

As an example, for Inference, we could have the implementation be backed by open source libraries like `[ torch | vLLM | TensorRT ]` as possible options.

A provider can also be just a pointer to a remote REST service -- for example, cloud providers or dedicated inference providers could serve these APIs.


## Llama Stack Distribution

A Distribution is where APIs and Providers are assembled together to provide a consistent whole to the end application developer. You can mix-and-match providers -- some could be backed by local code and some could be remote. As a hobbyist, you can serve a small model locally, but can choose a cloud provider for a large model. Regardless, the higher level APIs your app needs to work with don't need to change at all. You can even imagine moving across the server / mobile-device boundary as well always using the same uniform set of APIs for developing Generative AI applications.

## Supported Llama Stack Implementations
### API Providers


| **API Provider Builder** |    **Environments**    |     **Agents**     |   **Inference**    |     **Memory**     |     **Safety**     |   **Telemetry**    |
| :----------------------: | :--------------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|      Meta Reference      |      Single Node       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|        Fireworks         |         Hosted         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
|       AWS Bedrock        |         Hosted         |                    | :heavy_check_mark: |                    | :heavy_check_mark: |                    |
|         Together         |         Hosted         | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: |                    |
|          Ollama          |      Single Node       |                    | :heavy_check_mark: |                    |                    |
|           TGI            | Hosted and Single Node |                    | :heavy_check_mark: |                    |                    |
|          Chroma          |      Single Node       |                    |                    | :heavy_check_mark: |                    |                    |
|        PG Vector         |      Single Node       |                    |                    | :heavy_check_mark: |                    |                    |
|    PyTorch ExecuTorch    |     On-device iOS      | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
|          Qdrant          | Hosted and Single Node |                    |                    | :heavy_check_mark: |                    |                    |


### Distributions
|  **Distribution Provider** |  **Docker** | **Inference** | **Memory** | **Safety** | **Telemetry** |
| :----: | :----: | :----: | :----: | :----: | :----: |
|  Meta Reference |  [Local GPU](https://hub.docker.com/repository/docker/llamastack/llamastack-local-gpu/general), [Local CPU](https://hub.docker.com/repository/docker/llamastack/llamastack-local-cpu/general) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|  Dell-TGI | [Local TGI + Chroma](https://hub.docker.com/repository/docker/llamastack/llamastack-local-tgi-chroma/general)  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |


## Installation

You can install this repository as a [package](https://pypi.org/project/llama-stack/) with `pip install llama-stack`

If you want to install from source:

```bash
mkdir -p ~/local
cd ~/local
git clone git@github.com:meta-llama/llama-stack.git

conda create -n stack python=3.10
conda activate stack

cd llama-stack
$CONDA_PREFIX/bin/pip install -e .
```

## Documentations

The `llama` CLI makes it easy to work with the Llama Stack set of tools. Please find the following docs for details.

* [CLI reference](docs/cli_reference.md)
    * Guide using `llama` CLI to work with Llama models (download, study prompts), and building/starting a Llama Stack distribution.
* [Getting Started](docs/getting_started.md)
    * Guide to build and run a Llama Stack server.
* [Contributing](CONTRIBUTING.md)


## Llama Stack Client SDK

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) |
| Node   | [llama-stack-client-node](https://github.com/meta-llama/llama-stack-client-node) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) |

Check out our client SDKs for connecting to Llama Stack server in your preferred language, you can choose from [python](https://github.com/meta-llama/llama-stack-client-python), [node](https://github.com/meta-llama/llama-stack-client-node), [swift](https://github.com/meta-llama/llama-stack-client-swift), and [kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) programming languages to quickly build your applications.
