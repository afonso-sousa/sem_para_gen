import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from train import (
    DatasetArguments,
    DataTrainingArguments,
    ModelArguments,
    processing_function_wrapper,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationArguments:
    num_return_sequences: int = field(
        default=1,
        metadata={"help": "Number of sequences to generate."},
    )
    beam_width: int = field(
        default=3,
        metadata={"help": "Number of beam groups."},
    )
    num_beam_groups: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beam groups."},
    )
    repetition_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "The parameter for repetition penalty."},
    )
    diversity_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "The parameter for diversity penalty."},
    )
    early_stopping: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to stop the beam search when at least \
                num_beams sentences are finished per batch or not."
            )
        },
    )
    penalty_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "The degeneration penalty for contrastive search."},
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "The size of the candidate set that is used to re-rank for contrastive search."
        },
    )


def main():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DatasetArguments,
            DataTrainingArguments,
            Seq2SeqTrainingArguments,
            GenerationArguments,
        )
    )
    (
        model_args,
        dataset_args,
        data_args,
        training_args,
        generate_args,
    ) = parser.parse_args_into_dataclasses()

    training_args.do_train = False
    training_args.do_evaluate = False
    training_args.do_predict = True
    training_args.predict_with_generate = True

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO
        if is_main_process(training_args.local_rank)
        else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {}
    if dataset_args.test_file is not None:
        data_files["test"] = dataset_args.test_file
    else:
        raise ValueError("Please provide a test file.")
    processed_datasets = load_dataset(
        dataset_args.dataset_name, data_files=data_files
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
    )
    if model_args.model_name_or_path is not None:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path
        )
    else:
        raise ValueError("You must specify model_name_or_path or config_name")

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = processed_datasets["test"].column_names

    # Get the column names for input/target.
    if data_args.source_column is None:
        source_column = column_names[0]
    else:
        source_column = data_args.source_column
        if source_column not in column_names:
            raise ValueError(
                f"--source_column' value '{data_args.source_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.target_column is None:
        target_column = column_names[1]
    else:
        target_column = data_args.target_column
        if target_column not in column_names:
            raise ValueError(
                f"--target_column' value '{data_args.target_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    if training_args.label_smoothing_factor > 0 and not hasattr(
        model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    preprocess_function = processing_function_wrapper(
        source_column,
        tokenizer,
        data_args.max_source_length,
        max_target_length,
    )

    input_dataset = processed_datasets.map(
        preprocess_function,
        batched=False,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    label_pad_token_id = -100

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("*** Predict ***")

    # from datasets import Dataset

    predict_results = trainer.predict(
        # Dataset.from_dict(input_dataset["test"][10:20]),
        input_dataset["test"],
        max_length=data_args.val_max_target_length,
        penalty_alpha=generate_args.penalty_alpha,  # 0.6,
        top_k=generate_args.top_k,  # 6,
        # num_beams=generate_args.num_return_sequences * generate_args.beam_width,
        # num_return_sequences=generate_args.num_return_sequences,
        # num_beam_groups=generate_args.num_beam_groups,
        # repetition_penalty=generate_args.repetition_penalty,
        # diversity_penalty=generate_args.diversity_penalty,
        # early_stopping=generate_args.early_stopping,
    )

    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            preds = predict_results.predictions
            # Replace -100s used for padding as we can't decode them
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(
                preds,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(
                training_args.output_dir, data_args.output_file_name
            )
            result = pd.DataFrame(
                {
                    "source": processed_datasets["test"][source_column],
                    "target": processed_datasets["test"][target_column],
                    "prediction": predictions,
                }
            )
            result.to_csv(output_prediction_file, index=False, sep="\t")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
