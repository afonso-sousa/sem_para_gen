"""
Evaluation script.
"""

import argparse
import json
import logging
import os

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from bart_types import TokenBartForConditionalGeneration
from bart_types_collator import DataCollatorForSeq2SeqWithCustomTypes
from utils import (
    parse_amr_into_reduced_form,
    processing_function_for_AMR,
    processing_function_for_dep_trees,
    processing_function_for_semantic_graph,
    standard_processing_function,
)

logger = get_logger(__name__)


# Parsing input arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model for paraphrase generation"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use.",
    )
    parser.add_argument(
        "--splits_suffix",
        type=str,
        default=None,
        help="The suffix of the dataset splits.",
    )
    parser.add_argument(
        "--with_graph",
        action="store_true",
        help="Whether to use append linearized semantic graph as input",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="graph",
        choices=["graph", "dp", "amr"],
        help="Specify the graph type (semantic graph, dependency parsing or amr)",
    )
    parser.add_argument(
        "--with_token_types",
        action="store_true",
        help="Whether to embed token label information",
    )
    parser.add_argument(
        "--reentrancy_tokens",
        action="store_true",
        help="Whether to not use reentrancy tokens",
    )
    parser.add_argument(
        "--delimiters",
        action="store_true",
        help="Whether to not use delimiter tokens",
    )
    parser.add_argument(
        "--relations",
        action="store_true",
        help="Whether to not use delimiter tokens",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Truncate the number of evaluation examples to this value if set.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--scores_file",
        type=str,
        default="eval_results.json",
        help="Where to store the final scores.",
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        default=None,
        help="Where to store the final predictions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=3,
        help="Number of beam groups.",
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=None,
        help="Number of beam groups.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="The parameter for repetition penalty.",
    )
    parser.add_argument(
        "--diversity_penalty",
        type=float,
        default=None,
        help="The parameter for diversity penalty.",
    )
    args = parser.parse_args()

    # Sanity checks

    if args.dataset_name is None:
        raise ValueError("Make sure to provide a dataset name")

    if args.model_name_or_path is None:
        raise ValueError("Make sure to provide a model name")

    if args.with_token_types is not None and args.with_graph is None:
        raise ValueError(
            "Cannot set `with_token_types` without setting `with_graph` to True."
        )

    if args.reentrancy_tokens is not None and args.with_graph is None:
        raise ValueError(
            "Cannot set `reentrancy_tokens` without setting `with_graph` to True."
        )

    if args.delimiters is not None and args.with_graph is None:
        raise ValueError(
            "Cannot set `delimiters` without setting `with_graph` to True."
        )

    if args.relations is not None and args.with_graph is None:
        raise ValueError("Cannot set `relations` without setting `with_graph` to True.")

    return args


def main():
    # Parse the arguments
    args = parse_args()

    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    raw_datasets = load_dataset(
        args.dataset_name,
        data_files={
            "test": f"test_{args.splits_suffix}.jsonl",
        },
    )

    if args.graph_type == "amr":
        with accelerator.main_process_first():
            raw_datasets = raw_datasets.map(
                parse_amr_into_reduced_form,
                batched=False,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc="Parsing AMR into reduced form",
            )

    config = AutoConfig.from_pretrained(args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )

    if args.with_token_types:
        if args.model_name_or_path != "facebook/bart-base":
            raise ValueError(
                "Token types are only supported for models other than BART at the moment."
            )
        model = TokenBartForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            config=config,
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["test"].column_names

    # Temporarily set max_target_length for training.
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length

    if args.with_graph:
        if args.graph_type == "graph":
            preprocess_function = processing_function_for_semantic_graph(
                tokenizer,
                max_source_length,
                max_target_length,
                with_token_types=args.with_token_types,
                reentrancy_tokens=args.reentrancy_tokens,
                delimiters=args.delimiters,
                relations=args.relations,
            )
        elif args.graph_type == "dp":
            preprocess_function = processing_function_for_dep_trees(
                tokenizer,
                max_source_length,
                max_target_length,
            )
        elif args.graph_type == "amr":
            preprocess_function = processing_function_for_AMR(
                tokenizer,
                max_source_length,
                max_target_length,
            )
    else:
        preprocess_function = standard_processing_function(
            tokenizer,
            max_source_length,
            max_target_length,
        )

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=not args.with_graph,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    eval_dataset = processed_datasets["test"]

    if args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # DataLoaders creation:
    label_pad_token_id = -100

    # this collator supports the raw input
    data_collator = DataCollatorForSeq2SeqWithCustomTypes(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        # drop_last=True,
    )

    # Prepare everything with `accelerator`.
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    # metric = evaluate.load("sacrebleu")
    metric = evaluate.load("metrics/my_metric")

    def postprocess_text(preds, labels, input_ids=None):
        preds = [pred.strip() for pred in preds]
        if input_ids:
            labels = [label.strip() for label in labels]
            input_ids = [ids.strip() for ids in input_ids]
            return preds, labels, input_ids

        labels = [[label.strip()] for label in labels]

        return preds, labels

    total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}"
    )
    logger.info(
        f"  Total eval batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )

    model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": (
            args.val_max_target_length if args is not None else config.max_length
        ),
        "num_beams": args.num_beams,
    }
    if args.num_return_sequences > 1:
        more_gen_kwargs = {
            "num_beams": args.num_return_sequences * args.beam_width,
            "num_return_sequences": args.num_return_sequences,
            "num_beam_groups": args.num_beam_groups,
            "repetition_penalty": args.repetition_penalty,
            "diversity_penalty": args.diversity_penalty,  # higher the penalty, the more diverse are the outputs
        }
        gen_kwargs = {**gen_kwargs, **more_gen_kwargs}

    samples_seen = 0
    references = []
    predictions = []
    sources = []
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            if args.with_token_types:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_label_ids=batch["token_label_ids"],
                    **gen_kwargs,
                )
            else:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()

            # If we did not pad to max length, we need to pad the labels too
            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )

            labels = accelerator.gather(labels).cpu().numpy()

            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            # Expand labels to match the size of generated_tokens
            expanded_labels = np.repeat(
                labels, len(generated_tokens) // len(labels), axis=0
            )

            # input_ids = accelerator.pad_across_processes(
            #     batch["raw_inputs"] if args.with_graph else batch["input_ids"],
            #     dim=1,
            #     pad_index=tokenizer.pad_token_id,
            # )

            # input_ids = accelerator.gather(input_ids).cpu().numpy()

            # # Expand labels to match the size of generated_tokens
            # expanded_input_ids = np.repeat(
            #     input_ids, len(generated_tokens) // len(input_ids), axis=0
            # )

            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(
                expanded_labels, skip_special_tokens=True
            )

            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]

            # decoded_input_ids = tokenizer.batch_decode(
            #     expanded_input_ids, skip_special_tokens=True
            # )

            # (
            #     decoded_preds,
            #     decoded_labels,
            #     decoded_input_ids,
            # ) = postprocess_text(decoded_preds, decoded_labels, decoded_input_ids)

            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    decoded_preds = decoded_preds[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                    decoded_labels = decoded_labels[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                    # decoded_input_ids = decoded_input_ids[
                    #     : len(eval_dataloader.dataset) - samples_seen
                    # ]
                else:
                    samples_seen += len(decoded_labels)

            original_inputs = raw_datasets["test"]["source"][
                step * total_batch_size : (step + 1) * total_batch_size
            ]

            references.extend(decoded_labels)
            predictions.extend(decoded_preds)
            sources.extend(original_inputs)

            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
                sources=original_inputs,
            )
    eval_metric = metric.compute()
    logger.info({"bleu": eval_metric["bleu"]})
    logger.info({"self_bleu": eval_metric["self_bleu"]})
    logger.info({"ibleu": eval_metric["ibleu"]})
    logger.info({"sbert": f"{eval_metric['sbert_mean']} +- {eval_metric['sbert_std']}"})
    logger.info({"rougeL": f"{eval_metric['rougeL']}"})
    logger.info(
        {
            "para_score": f"{eval_metric['para_score_mean']} +- {eval_metric['para_score_std']}"
        }
    )

    if args.output_dir is not None:
        if args.scores_file:
            with open(os.path.join(args.output_dir, args.scores_file), "w") as f:
                json.dump(
                    {
                        "bleu": eval_metric["bleu"],
                        "self_bleu": eval_metric["self_bleu"],
                        "ibleu": eval_metric["ibleu"],
                        "sbert": f"{eval_metric['sbert_mean']} +- {eval_metric['sbert_std']}",
                        "rougeL": f"{eval_metric['rougeL']}",
                        "para_score": f"{eval_metric['para_score_mean']} +- {eval_metric['para_score_std']}",
                    },
                    f,
                )

        if args.predictions_file:
            result = pd.DataFrame(
                {
                    "source": sources,  # raw_eval_dataset["source"],
                    "target": references,  # raw_eval_dataset["target"],
                    "prediction": predictions,
                }
            )
            result.to_csv(
                os.path.join(args.output_dir, args.predictions_file),
                index=False,
                sep="\t",
            )


if __name__ == "__main__":
    main()
