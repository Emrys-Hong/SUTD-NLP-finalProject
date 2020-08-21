from config import Config
import torch.nn as nn
from typing import Tuple, List, Dict, Union
import torch
from transformers import PreTrainedTokenizer, AdamW, get_linear_schedule_with_warmup
import collections
import numpy as np
import json
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


Feature = collections.namedtuple('Feature', 'input_ids attention_mask token_type_ids label_id')
Feature.__new__.__defaults__ = (None,) * 4


class NERDataset(Dataset):

    def __init__(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizer,
                 label_map: Dict[str, int] = None,
                 test_strings: List[str] = None) -> None:
        instances = []
        if file is not None:
            data = read_data(file, 0)
            labels = read_data(file, 1)
            for sentence, label in zip(data, labels):
                instances.append(Instance(sentence=sentence, labels=label))
        else:
            for sentence in test_strings:
                instances.append(Instance(sentence=sentence))
        self.instances = instances
        self._features = convert_instances_to_feature_tensors(instances=instances,
                                                              tokenizer=tokenizer,
                                                              label_map=label_map,
                                                              sep_token=tokenizer.sep_token,
                                                              cls_token=tokenizer.cls_token,
                                                              pad_token_id=tokenizer.pad_token_id,
                                                              pad_token_segment_id=tokenizer.pad_token_type_id)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx) -> Feature:
        return self._features[idx]


class Instance:
    def __init__(self, sentence: List[str], labels: List[str] = None) -> None:
        self._sentence = sentence
        self._labels = labels


def convert_instances_to_feature_tensors(instances: List[Instance],
                                         tokenizer: PreTrainedTokenizer,
                                         label_map: Dict[str, int],
                                         sep_token: str = "[SEP]",
                                         cls_token: str = "[CLS]",
                                         pad_token_id: int = 0,
                                         pad_token_segment_id: int = 0) -> List[Feature]:
    features = []
    for idx, inst in enumerate(instances):
        sentence = inst._sentence
        labels = inst._labels
        label_id = [label_map['O']] + [label_map[label] for label in labels] + [label_map['O']] if labels is not None else -100
        wordpiece_tokens = [cls_token] + sentence + [sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(wordpiece_tokens)
        segment_ids = [0] * (len(sentence) + 2)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) == len(segment_ids)
        features.append(Feature(input_ids=input_ids,
                                attention_mask=input_mask,
                                token_type_ids=segment_ids,
                                label_id=label_id))
    return features


def collate_cloze_test(batch: List[Feature], tokenizer:PreTrainedTokenizer):
    max_wordpiece_length = max([len(feature.input_ids) for feature in batch])
    for i, feature in enumerate(batch):
        padding_length = max_wordpiece_length - len(feature.input_ids)
        input_ids = feature.input_ids + [tokenizer.pad_token_id] * padding_length
        mask = feature.attention_mask + [0] * padding_length
        type_ids = feature.token_type_ids + [tokenizer.pad_token_type_id] * padding_length
        label_id = feature.label_id + [0] * padding_length

        assert len(input_ids) == max_wordpiece_length
        assert len(mask) == max_wordpiece_length
        assert len(type_ids) == max_wordpiece_length
        batch[i] = Feature(input_ids=np.asarray(input_ids), attention_mask=np.asarray(mask), token_type_ids=np.asarray(type_ids), label_id=np.asarray(label_id))

    results = Feature(*(default_collate(samples) for samples in zip(*batch)))
    return results


def read_data(path, column=0):
    """column=0 means input sequence, column=1 means label
    """
    with open(path) as f:
        lines = f.readlines()

    data = []
    sample = []

    for line in lines:
        formatted_line = line.strip()

        if len(formatted_line) > 0:
            split_data = formatted_line.split(" ")
            sample.append(split_data[column])

        else:
            data.append(sample)
            sample = []

    return data


def get_optimizers(config: Config, model: nn.Module, num_training_steps: int, weight_decay: float = 0.01) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    no_decay = ["bias", "LayerNorm.weight", 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=1e-6, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def write_data(file:str, data) -> None:
    with open(file, "w", encoding="utf-8") as write_file:
        # json_results = json.dumps(results)
        # print(json_results)
        json.dump(data, write_file, indent=4)
