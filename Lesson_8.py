import os

from datasets import load_dataset
from dotenv import load_dotenv
from predibase import (GRPOConfig, Predibase, RewardFunctionsConfig, RewardFunctionsRuntimeConfig,
                        SamplingParamsConfig, SFTConfig)
from reward_functions import (guess_value, output_format_check, uses_previous_feedback)

## ------------------------------------------------------ ##
load_dotenv("../.env")
pb = Predibase(api_token = os.environ["PREDIBASE_API_KEY"])

## ------------------------------------------------------ ##
dataset = load_dataset("predibase/wordle-grpo", split = "train")
dataset = dataset.to_pandas()

try:
    dataset = pb.datasets.from_pandas_dataframe(
                                                dataset,
                                                name = "wordle_grpo_data"
                                                )

except Exception:
    dataset = pb.datasets.get("wordle_grpo_data")

## ------------------------------------------------------ ##
repo = pb.repos.create(name = "wordle", exists_ok = True)

## ------------------------------------------------------ ##
pb.finetuning.jobs.create(
                config = GRPOConfig(
                        base_model = "qwen2-5-7b-instruct",
                        reward_fns = RewardFunctionsConfig(
                                    runtime=RewardFunctionsRuntimeConfig(
                                                                        packages = ["pandas"]
                                                                        ),
                                    functions = {
                                                "output_format_check" : output_format_check,
                                                "uses_previous_feedback" : uses_previous_feedback,
                                                "guess_value" : guess_value,
                                                }
                                    ),
                        sampling_params = SamplingParamsConfig(max_tokens = 4096),
                        num_generations = 16
                        ),
                dataset = dataset,
                repo = "wordle",
                description = "Wordle GRPO"
                )

## ------------------------------------------------------ ##
dataset = load_dataset("predibase/wordle-sft", split = "train")
dataset = dataset.to_pandas()

dataset = pb.datasets.from_pandas_dataframe(dataset, name = "wordle_sft_data")

repo = pb.repos.create(name = "wordle", exists_ok = True)

pb.finetuning.jobs.create(
                        config = SFTConfig(
                                        base_model = "qwen2-5-7b-instruct",
                                        epochs = 10,
                                        rank = 64,
                                        target_modules = ["q_proj", "k_proj",
                                                        "v_proj", "o_proj",
                                                        "gate_proj", "down_proj",
                                                        "up_proj"],
                                        ),
                        dataset = dataset,
                        repo = "wordle",
                        description = "Wordle SFT, 10 epochs"
                        )

## ------------------------------------------------------ ##
dataset = pb.datasets.get("wordle_grpo_data")

pb.finetuning.jobs.create(
                    config = GRPOConfig(
                            base_model = "qwen2-5-7b-instruct",
                            reward_fns = RewardFunctionsConfig(
                                    runtime = RewardFunctionsRuntimeConfig(packages = ["pandas"]),
                                    functions = {
                                                "output_format_check" : output_format_check,
                                                "uses_previous_feedback" : uses_previous_feedback,
                                                "guess_value" : guess_value,
                                                }
                                    ),
                            epochs = 3,
                            enable_early_stopping = False,
                            sampling_params = SamplingParamsConfig(max_tokens = 4096),
                            num_generations = 8
                            ),
                    continue_from_version = "wordle/1",
                    dataset = dataset,
                    repo = "wordle",
                    description = "Wordle GRPO"
                    )
