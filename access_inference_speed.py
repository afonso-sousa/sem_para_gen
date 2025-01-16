import random
import statistics
import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

from bart_types import TokenBartForConditionalGeneration
from bart_types_collator import DataCollatorForSeq2SeqWithCustomTypes
from glossary import EXTRA_TOKENS, SEM_DEPS
from utils import processing_function_for_semantic_graph, standard_processing_function


def load_model(model_path, with_token_types=False):
    config = AutoConfig.from_pretrained(
        model_path,
    )
    if with_token_types:
        model = TokenBartForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            config=config,
        )
    model.eval()
    return model


def perform_inference(model, batch):
    with torch.no_grad():
        output = model(**batch)
    return output


# model_path = "output/facebook/bart-base-qqppos-lr1e-4-standard"
model_path = "output/facebook/bart-base-qqppos-lr1e-4-with_tokens"

with_graph = True
with_token_types = True
print(f"With Graph: {with_graph}")
print(f"With Token Type: {with_token_types}")
model = load_model(model_path, with_token_types)

# batch_size = 3
# max_seq_length = 30

max_source_length = 128
max_target_length = 64

special_tokens = EXTRA_TOKENS + SEM_DEPS if with_graph else []

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    extra_ids=0,  # no need for sentinel tokens
    additional_special_tokens=special_tokens,
    use_fast=True,
    add_prefix_space=True,
)

preprocess_function = (
    processing_function_for_semantic_graph(
        tokenizer,
        max_source_length,
        max_target_length,
        with_token_types=with_token_types,
        reentrancy_tokens=True,
    )
    if with_graph
    else standard_processing_function(
        tokenizer,
        max_source_length,
        max_target_length,
    )
)

raw_datasets = load_dataset(
    "processed-data/qqppos",
    data_files={
        "validation": "validation_with_graph.jsonl",
    },
)

column_names = raw_datasets["validation"].column_names

random.seed(10)
random_idx = random.sample(range(len(raw_datasets["validation"])), 1)


if with_token_types:
    data_collator = DataCollatorForSeq2SeqWithCustomTypes(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
    )
else:
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

# Set the number of inference iterations
num_iterations = 1000  # Adjust as needed based on your testing requirements

# Measure inference time
inference_times = []
processing_times = []

for _ in tqdm(range(num_iterations)):
    start_time = time.time()
    processed_dataset = (
        raw_datasets["validation"]
        .select(random_idx)
        .map(
            preprocess_function,
            batched=not with_graph,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
    )
    eval_dataloader = DataLoader(
        processed_dataset,
        collate_fn=data_collator,
        batch_size=1,
    )
    batch = next(iter(eval_dataloader))
    end_time = time.time()
    processing_time = end_time - start_time
    processing_times.append(processing_time)

    start_time = time.time()
    output = perform_inference(model, batch)
    end_time = time.time()
    inference_time = end_time - start_time
    inference_times.append(inference_time)

# Calculate average processing time
average_processing_time = sum(processing_times) / num_iterations

# Calculate average inference time
average_inference_time = sum(inference_times) / num_iterations

# Calculate standard deviation of processing and inference times
processing_std_deviation = statistics.stdev(processing_times) / num_iterations
inference_std_deviation = statistics.stdev(inference_times) / num_iterations

print(
    f"Average Processing Time for {num_iterations} iterations: {average_processing_time:.6f} seconds"
)
print(
    f"Average Inference Time for {num_iterations} iterations: {average_inference_time:.6f} seconds"
)
print(f"Standard Deviation of Processing Time: {processing_std_deviation:.6f} seconds")
print(f"Standard Deviation of Inference Time: {inference_std_deviation:.6f} seconds")
