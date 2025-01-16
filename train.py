"""
Fine-tuning the library models for sequence to sequence.
"""

import argparse
import json
import logging
import math
import os
import random
import sys

import datasets
import evaluate
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)

from bart_types import TokenBartForConditionalGeneration
from bart_types_collator import DataCollatorForSeq2SeqWithCustomTypes
from glossary import EXTRA_TOKENS, SEM_DEPS, SYN_DEPS
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
        description="Finetune a transformers model on a text classification task"
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
        default=1024,
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
        "--freeze_embeddings",
        action="store_true",
        help="Whether to freeze the embedding layers' parameters.",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Whether to freeze the encoder parameters.",
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
        help="Path to pretrained model or model identifier.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Truncate the number of training examples to this value if set.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Truncate the number of evaluation examples to this value if set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--evaluation_interval",
        type=int,
        default=1,
        help="Evaluate every X epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None:
        raise ValueError("Need training/validation data.")

    if args.model_name_or_path is None:
        raise ValueError("Please provide a model name")

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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
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
            "train": f"train_{args.splits_suffix}.jsonl",
            "validation": f"validation_{args.splits_suffix}.jsonl",
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

    if args.with_graph:
        if args.graph_type == "graph":
            special_tokens = EXTRA_TOKENS + SEM_DEPS
        elif args.graph_type == "dp":
            special_tokens = list(set(f":{dep}" for dep in SYN_DEPS.keys()))
        elif args.graph_type == "amr":
            special_tokens = list(
                {
                    el
                    for node_tokens in raw_datasets["train"]["node_tokens"]
                    if node_tokens is not None
                    for el in node_tokens
                    if el.startswith(":") and len(el) > 1
                }
            )
        else:
            raise ValueError("Unknown graph type")
    else:
        special_tokens = []

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        extra_ids=0,  # no need for sentinel tokens
        additional_special_tokens=special_tokens,
        use_fast=not args.use_slow_tokenizer,
        add_prefix_space=True,
    )
    if args.with_token_types:
        if args.model_name_or_path != "facebook/bart-base":
            raise ValueError(
                "Token types are only supported for models other than BART at the moment."
            )
        from glossary import UPOS

        config.type_vocab_size = 3 + len(
            UPOS
        )  # 3 is pad (0), special (1) and relation (2)
        model = TokenBartForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            config=config,
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    column_names = raw_datasets["train"].column_names

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

    # Malformed entries will have None values
    processed_datasets = processed_datasets.filter(
        lambda example: all(value is not None for value in example.values())
    )

    # def retrieve_stats(examples):
    #     import re
    #     from collections import Counter

    #     from semantic_graph import Graph

    #     output_dict = {}

    #     g = Graph(nodes=examples["nodes"], edges=examples["edges"])
    #     linearized_token_list = g.linearize(
    #         reentrancy_tokens=False,
    #         delimiters=False,
    #         relations=False,
    #     )
    #     linearized_types, linearized_tokens = list(zip(*linearized_token_list))

    #     pattern = r"<R\d+>"
    #     text = " ".join(linearized_tokens)
    #     matches = re.findall(pattern, text)
    #     output_dict["len_Rn"] = len(matches)

    #     occurrences_counter = Counter(matches)
    #     output_dict["has_reentrancy"] = any(
    #         count > 1 for count in occurrences_counter.values()
    #     )

    #     unique_relations = {
    #         relation
    #         for relation in linearized_tokens
    #         if relation.startswith(":")
    #     }
    #     output_dict["unique_relations"] = set(unique_relations)

    #     output_dict["len"] = len(linearized_tokens)
    #     output_dict["unique_types"] = set(linearized_types)

    #     return output_dict

    # retrieved_stats = raw_datasets["train"].map(
    #     retrieve_stats,
    #     batched=not args.with_graph,
    #     num_proc=args.preprocessing_num_workers,
    #     remove_columns=column_names,
    #     load_from_cache_file=not args.overwrite_cache,
    #     desc="Running tokenizer on dataset",
    # )

    # import statistics

    # len_Rn_values = [item["len_Rn"] for item in retrieved_stats]
    # len_values = [item["len"] for item in retrieved_stats]
    # num_unique_types_values = [
    #     len(item["unique_types"]) for item in retrieved_stats
    # ]
    # types_values = [
    #     value for item in retrieved_stats for value in item["unique_types"]
    # ]
    # relations_values = [
    #     value for item in retrieved_stats for value in item["unique_relations"]
    # ]
    # count_entries_with_reentrancy = sum(
    #     item["has_reentrancy"] for item in retrieved_stats
    # )

    # # Print the means
    # print("Mean len_Rn:", statistics.mean(len_Rn_values))
    # print("Max len_Rn:", max(len_Rn_values))
    # print("Mean len:", statistics.mean(len_values))
    # print("Mean num_unique_types:", statistics.mean(num_unique_types_values))
    # print("Total num unique_types:", len(set(types_values)))
    # print("Total num unique_relations:", len(set(relations_values)))
    # print("Num entries with reentrancy:", count_entries_with_reentrancy)
    # breakpoint()

    # from collections import Counter

    # flat_list = [
    #     item
    #     for sublist in processed_datasets["train"]["input_ids"]
    #     for item in sublist
    # ]
    # counts = Counter(flat_list)
    # c = sum(
    #     counts[item] for item in tokenizer.convert_tokens_to_ids(EXTRA_TOKENS)
    # )

    import statistics

    mean = statistics.mean(
        [len(sublist) for sublist in processed_datasets["train"]["input_ids"]]
    )
    stdev = statistics.stdev(
        [len(sublist) for sublist in processed_datasets["train"]["input_ids"]]
    )
    print(f"{mean} +- {stdev}")

    train_dataset = processed_datasets["train"]
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = processed_datasets["validation"]
    if args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    label_pad_token_id = -100

    if args.with_token_types:
        data_collator = DataCollatorForSeq2SeqWithCustomTypes(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config[
                "lr_scheduler_type"
            ].value
            accelerator.init_trackers("translation_no_trainer", experiment_config)

    # Metric
    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        if epoch % args.evaluation_interval == 0:
            model.eval()
            if args.val_max_target_length is None:
                args.val_max_target_length = args.max_target_length

            gen_kwargs = {
                "max_length": (
                    args.val_max_target_length
                    if args is not None
                    else config.max_length
                ),
                "num_beams": args.num_beams,
            }
            samples_seen = 0
            for step, batch in tqdm(
                enumerate(eval_dataloader), total=len(eval_dataloader)
            ):
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
                        generated_tokens,
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )
                    labels = batch["labels"]

                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(
                        batch["labels"],
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )

                    generated_tokens = (
                        accelerator.gather(generated_tokens).cpu().numpy()
                    )
                    labels = accelerator.gather(labels).cpu().numpy()

                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                    decoded_preds = tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    decoded_labels = tokenizer.batch_decode(
                        labels, skip_special_tokens=True
                    )

                    decoded_preds, decoded_labels = postprocess_text(
                        decoded_preds, decoded_labels
                    )

                    # If we are in a multiprocess environment, the last batch has duplicates
                    if accelerator.num_processes > 1:
                        if step == len(eval_dataloader) - 1:
                            decoded_preds = decoded_preds[
                                : len(eval_dataloader.dataset) - samples_seen
                            ]
                            decoded_labels = decoded_labels[
                                : len(eval_dataloader.dataset) - samples_seen
                            ]
                        else:
                            samples_seen += len(decoded_labels)

                    metric.add_batch(
                        predictions=decoded_preds, references=decoded_labels
                    )
            eval_metric = metric.compute()
            logger.info({"bleu": eval_metric["score"]})

            if args.with_tracking:
                accelerator.log(
                    {
                        "bleu": eval_metric["score"],
                        "train_loss": total_loss.item() / len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"eval_bleu": eval_metric["score"]}, f)


if __name__ == "__main__":
    main()
