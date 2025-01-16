from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from transformers import DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class DataCollatorForSeq2SeqWithCustomTypes(DataCollatorForSeq2Seq):
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        token_label_ids = (
            [feature["token_label_ids"] for feature in features]
            if "token_label_ids" in features[0].keys()
            else None
        )

        raw_input_ids = (
            [feature["raw_inputs"] for feature in features]
            if "raw_inputs" in features[0].keys()
            else None
        )

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        # Pad token_label_ids if present
        if token_label_ids is not None:
            max_token_type_length = max(len(l) for l in token_label_ids)
            if self.pad_to_multiple_of is not None:
                max_token_type_length = (
                    (max_token_type_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            for feature in features:
                remainder = [0] * (
                    max_token_type_length - len(feature["token_label_ids"])
                )
                if isinstance(feature["token_label_ids"], list):
                    feature["token_label_ids"] = (
                        feature["token_label_ids"] + remainder
                        if padding_side == "right"
                        else remainder + feature["token_label_ids"]
                    )
                elif padding_side == "right":
                    feature["token_label_ids"] = np.concatenate(
                        [feature["token_label_ids"], remainder]
                    ).astype(np.int64)
                else:
                    feature["token_label_ids"] = np.concatenate(
                        [remainder, feature["token_label_ids"]]
                    ).astype(np.int64)

        if raw_input_ids is not None:
            max_raw_inputs_length = max(len(l) for l in raw_input_ids)
            if self.pad_to_multiple_of is not None:
                max_raw_inputs_length = (
                    (max_raw_inputs_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            for feature in features:
                remainder = [0] * (
                    max_raw_inputs_length - len(feature["raw_inputs"])
                )
                if isinstance(feature["raw_inputs"], list):
                    feature["raw_inputs"] = (
                        feature["raw_inputs"] + remainder
                        if padding_side == "right"
                        else remainder + feature["raw_inputs"]
                    )
                elif padding_side == "right":
                    feature["raw_inputs"] = np.concatenate(
                        [feature["raw_inputs"], remainder]
                    ).astype(np.int64)
                else:
                    feature["raw_inputs"] = np.concatenate(
                        [remainder, feature["raw_inputs"]]
                    ).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = (
                self.model.prepare_decoder_input_ids_from_labels(
                    labels=features["labels"]
                )
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features
