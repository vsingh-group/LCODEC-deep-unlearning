# Code modified from https://github.com/dtuggener/LEDGAR_provision_classification/

import itertools
import json
import numpy as np
import re

import torch
from torch.utils.data import TensorDataset

from typing import List, Union, Dict, DefaultDict, Tuple
from collections import defaultdict
from sklearn.model_selection import train_test_split



def multihot(labels, label_map):
    res = np.zeros(len(label_map))
    for lbl in labels:
        res[label_map[lbl]] = 1.0
    return res


class DonData(object):

    def __init__(self, path):
        self.don_data = split_corpus(path)
        self.all_lbls = list(sorted({
            label
            for lbls in itertools.chain(
                self.don_data.y_train,
                self.don_data.y_test,
                self.don_data.y_dev if self.don_data.y_dev is not None else []
            )
            for label in lbls
        }))
        self.label_map = {
            label: i
            for i, label in enumerate(self.all_lbls)
        }

        total = 0
        self.class_weights = np.zeros(len(self.label_map), dtype=np.float32)
        for sample in self.train():
            self.class_weights += sample['label']
            total += 1
        self.class_weights = total / (len(self.label_map) * self.class_weights)

    def train(self):
        return [{
            'txt': x,
            'label': multihot(lbls, self.label_map),
        } for x, lbls in zip(self.don_data.x_train, self.don_data.y_train)]

    def test(self):
        return [{
            'txt': x,
            'label': multihot(lbls, self.label_map),
        } for x, lbls in zip(self.don_data.x_test, self.don_data.y_test)]

    def dev(self):
        return [{
            'txt': x,
            'label': multihot(lbls, self.label_map),
        } for x, lbls in zip(self.don_data.x_dev, self.don_data.y_dev)]

def convert_examples_to_features(
        examples,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token_segment_id=0,
        sep_token_extra=False,
        pad_on_left=False,
        pad_token_segment_id=0,
        sequence_segment_id=0,
        mask_padding_with_zero=True,
):
    # / ! \ copy-pasted from https://github.com/huggingface/pytorch-transformers/blob/master/examples/utils_glue.py
    # should work with bert and distilbert, will return dataset directly
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    all_input_ids = []
    all_input_masks = []
    all_segment_ids = []
    all_label_ids = []
    for ex_ix, example in enumerate(examples):

        tokens = tokenizer.tokenize(example['txt'])

        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        if sep_token_extra:
            tokens += [sep_token]
        segment_ids = [sequence_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        pad_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * pad_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * pad_length) + input_mask
            segment_ids = ([pad_token_segment_id] * pad_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * pad_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * pad_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * pad_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        all_input_ids.append(input_ids)
        all_input_masks.append(input_mask)
        all_segment_ids.append(segment_ids)
        all_label_ids.append(example['label'])

    input_id_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    input_mask_tensor = torch.tensor(all_input_masks, dtype=torch.long)
    segment_id_tensor = torch.tensor(all_segment_ids, dtype=torch.long)
    label_id_tensor = torch.tensor(all_label_ids, dtype=torch.float)

    return TensorDataset(
        input_id_tensor,
        input_mask_tensor,
        segment_id_tensor,
        label_id_tensor,
    )

"""
@dataclass
class SplitDataSet:
    x_train: List[str]
    y_train: List[List[str]]
    x_test: List[str]
    y_test: List[List[str]]
    x_dev: Union[List[str], None]
    y_dev: Union[List[List[str]], None]
"""

# python3.6
class SplitDataSet:
    def __init__(self, x_train, y_train,
                 x_test, y_test,
                 x_dev=None, y_dev=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_dev = x_dev
        self.y_dev = y_dev


def split_corpus(corpus_file: str, use_dev: bool = True,
                 test_size: float = 0.2, dev_size: Union[float, None] = 0.1,
                 random_state: int = 42) -> SplitDataSet:
    x: List[str] = []
    y: List[List[str]] = []
    doc_ids: List[str] = []
    for line in open(corpus_file, encoding='utf-8'):
        labeled_provision = json.loads(line)
        x.append(labeled_provision['provision'])
        y.append(labeled_provision['label'])
        doc_ids.append(labeled_provision['source'])

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=test_size,
                                                        random_state=random_state)

    if use_dev:
        x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                                                          test_size=dev_size,
                                                          random_state=random_state)
    else:
        x_dev, y_dev = None, None

    dataset = SplitDataSet(x_train, y_train, x_test, y_test, x_dev, y_dev)
    return dataset


def evaluate_multilabels(y: List[List[str]], y_preds: List[List[str]],
                         do_print: bool = False) -> DefaultDict[str, Dict[str, float]]:
    """
    Print classification report with multilabels
    :param y: Gold labels
    :param y_preds: Predicted labels
    :param do_print: Whether to print results
    :return: Dict of scores per label and overall
    """
    # Label -> TP/FP/FN -> Count
    label_eval: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
    assert len(y) == len(y_preds), "List of predicted and gold labels are of unequal length"
    for y_true, y_pred in zip(y, y_preds):
        for label in y_true:
            if label in y_pred:
                label_eval[label]['tp'] += 1
            else:
                label_eval[label]['fn'] += 1
        for label in y_pred:
            if label not in y_true:
                label_eval[label]['fp'] += 1

    max_len = max([len(l) for l in label_eval.keys()])
    if do_print:
        print('\t'.join(['Label'.rjust(max_len, ' '), 'Prec'.ljust(4, ' '), 'Rec'.ljust(4, ' '), 'F1'.ljust(4, ' '),
                     'Support']))

    eval_results: DefaultDict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    all_f1, all_rec, all_prec = [], [], []
    for label in sorted(label_eval.keys()):
        cnts = label_eval[label]
        if not cnts['tp'] == 0:
            prec = cnts['tp'] / (cnts['tp'] + cnts['fp'])
            rec = cnts['tp'] / (cnts['tp'] + cnts['fn'])
            f1 = (2 * prec * rec) / (prec + rec)
        else:
            prec, rec, f1 = 0.00, 0.00, 0.00
        eval_results[label]['prec'] = prec
        eval_results[label]['rec'] = rec
        eval_results[label]['f1'] = f1
        eval_results[label]['support'] = cnts['tp'] + cnts['fn']
        all_f1.append(f1)
        all_rec.append(rec)
        all_prec.append(prec)
        if do_print:
            print('\t'.join([label.rjust(max_len, ' '),
                         ('%.2f' % round(prec, 2)).ljust(4, ' '),
                         ('%.2f' % round(rec, 2)).ljust(4, ' '),
                         ('%.2f' % round(f1, 2)).ljust(4, ' '), 
                         str(cnts['tp'] + cnts['fn']).rjust(5, ' ')
                         ]))

    eval_results['Macro']['prec'] = sum(all_prec) / len(all_prec)
    eval_results['Macro']['rec'] = sum(all_rec) / len(all_rec)
    if eval_results['Macro']['prec'] + eval_results['Macro']['rec'] == 0:
        eval_results['Macro']['f1'] = 0.0
    else:
        eval_results['Macro']['f1'] = (2 * eval_results['Macro']['prec'] * eval_results['Macro']['rec']) / \
                                  (eval_results['Macro']['prec'] + eval_results['Macro']['rec'])
    eval_results['Macro']['support'] = len(y)

    # Micro
    all_tp = sum(label_eval[label]['tp'] for label in label_eval)
    all_fp = sum(label_eval[label]['fp'] for label in label_eval)
    all_fn = sum(label_eval[label]['fn'] for label in label_eval)
    if all_fp == 0:
        eval_results['Micro']['prec'] = 0
        eval_results['Micro']['rec'] = 0
        eval_results['Micro']['f1'] = 0
    else:
        eval_results['Micro']['prec'] = all_tp / (all_tp + all_fp)
        eval_results['Micro']['rec'] = all_tp / (all_tp + all_fn)
        micro_prec = eval_results['Micro']['prec']
        micro_rec = eval_results['Micro']['rec']
        if micro_prec + micro_rec == 0:
            eval_results['Micro']['f1'] = 0.0
        else:
            eval_results['Micro']['f1'] = (2 * micro_rec * micro_prec) / (micro_rec + micro_prec)
    eval_results['Micro']['support'] = len(y)

    if do_print:
        print('Macro Avg. Rec:', round(eval_results['Macro']['rec'], 2))
        print('Macro Avg. Prec:', round(eval_results['Macro']['prec'], 2))
        print('Macro F1:', round(eval_results['Macro']['f1'], 2))
        print()
        print('Micro Avg. Rec:', round(eval_results['Micro']['rec'], 2))
        print('Micro Avg. Prec:',  round(eval_results['Micro']['prec'], 2))
        print('Micro F1:', round(eval_results['Micro']['f1'], 2))

    return eval_results