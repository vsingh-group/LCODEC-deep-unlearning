# Code modified from https://github.com/dtuggener/LEDGAR_provision_classification/

import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

from pytorch_transformers import (
    DistilBertConfig,
    DistilBertTokenizer,
)
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.modeling_distilbert import (
    DistilBertPreTrainedModel,
    DistilBertModel,
)

from sklearn.metrics import f1_score, classification_report

from tqdm import tqdm, trange
import numpy as np

from ledgar_utils import DonData, convert_examples_to_features, evaluate_multilabels

import copy
import pandas as pd
import os
from torch.utils.data import Subset
from tqdm import tqdm

import time

from data_utils import getDatasets
from nn_utils import manual_seed

from scrub_tools import scrubSample, inp_perturb
from grad_utils import getGradObjs, gradNorm
from data_utils import SubsetDataWrapper



class DistilBertForMultilabelSequenceClassification(DistilBertPreTrainedModel):

    def __init__(self, config):
        super(DistilBertForMultilabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        print("Num labels: ", self.num_labels)

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            head_mask=None,
            labels=None,
            class_weights=None,
    ):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def train(train_dataset, model, train_params, class_weights=None):
    # TODO: magic numbers, defaults in run_glue.py
    batch_size = train_params['batch_size']
    n_epochs = train_params['epochs']
    weight_decay = train_params['weight_decay']
    learning_rate = train_params['learning_rate']
    adam_epsilon = train_params['adam_epsilon']
    warmup_steps = train_params['warmup_steps']
    seed = train_params['seed']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_grad_norm = train_params['max_grad_norm']

    print('Train Set Size: ', len(train_dataset))
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
    )

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)

    no_decay = {'bias', 'LayerNorm.weight'}
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=adam_epsilon,
    )
    scheduler = WarmupLinearSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        t_total=len(train_dataloader) // n_epochs,
    )

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iter = trange(n_epochs, desc='Epoch')
    epoch = 0
    outString = 'trained_models/'+"Ledgar"+"_"+"distilbert"+"_run_" + str(seed) + '_epochs_' + str(n_epochs)
    for _ in train_iter:
        grad_bank = None 
        nsamps = 0
        epoch_iter = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iter):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                #'token_type_ids': batch[2],  # probably used for distilbert
                'labels': batch[3],
                'class_weights': class_weights,
            }

            logits = model(**inputs)

            loss_fct = nn.BCEWithLogitsLoss(
                reduction='mean',
                pos_weight=class_weights,
            )
            loss = loss_fct(logits, inputs['labels'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            if epoch >= n_epochs-2:
                batch_gradbank, param_bank = getGradObjs(model)
                if grad_bank is None:
                    grad_bank = batch_gradbank
                else:
                    for key in grad_bank.keys():
                        grad_bank[key] += batch_gradbank[key]
            
            nsamps += batch[3].shape[0]

        if epoch >= n_epochs-2:
            for key in grad_bank.keys():
                grad_bank[key] = grad_bank[key]/nsamps
            print(f'saving params at epoch {epoch}...')
            torch.save(param_bank, outString + f'_epoch_{epoch}_params.pt')
            print(f'saving gradients at epoch {epoch}...')
            torch.save(grad_bank, outString + f'_epoch_{epoch}_grads.pt')

        epoch += 1

    return global_step, tr_loss / global_step


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def evaluate(eval_dataset, model, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_loader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=batch_size
    )

    preds = None
    out_label_ids = None
    for batch in tqdm(eval_loader, desc="Evaluation"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                #'token_type_ids': batch[2],
                'labels': batch[3]
            }

            logits = model(**inputs)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids,
                    inputs['labels'].detach().cpu().numpy(),
                    axis=0,
                )

    return {
        'pred': sigmoid(preds),
        'truth': out_label_ids,
    }


def tune_threshs(probas, truth):
    res = np.zeros(probas.shape[1])

    assert np.alltrue(probas >= 0.0)
    assert np.alltrue(probas <= 1.0)

    for i in range(probas.shape[1]):
        if np.sum(truth[:, i]) > 4 :
            thresh = max(
                np.linspace(
                    0.0,
                    1.0,
                    num=100,
                ),
                key=lambda t: f1_score(y_true=truth[:, i], y_pred=(probas[:, i] > t), pos_label=1, average='binary')
            )
            res[i] = thresh
        else:
            # res[i] = np.max(probas[:, i])
            res[i] = 0.5

    return res


def apply_threshs(probas, threshs):
    res = np.zeros(probas.shape)

    for i in range(probas.shape[1]):
        res[:, i] = probas[:, i] > threshs[i]

    return res


def multihot_to_label_lists(label_array, label_map):
    label_id_to_label = {
        v: k
        for k, v in label_map.items()
    }
    res = []
    for i in range(label_array.shape[0]):
        lbl_set = []
        for j in range(label_array.shape[1]):
            if label_array[i, j] > 0:
                lbl_set.append(label_id_to_label[j])
        res.append(lbl_set)
    return res


def subsample(data, quantile, n_classes):
    class_counts = np.zeros(n_classes, dtype=np.int32)
    for sample in data:
        class_counts += (sample['label'] > 0)

    cutoff = int(np.quantile(class_counts, q=quantile))

    n_to_sample = np.minimum(class_counts, cutoff)

    index_map = {
        i: []
        for i in range(n_classes)
    }
    to_keep = set()
    for ix, sample in enumerate(data):
        if np.sum(sample['label']) > 1:
            to_keep.add(ix)
            n_to_sample -= (sample['label'] > 0)
        else:
            label = np.argmax(sample['label'])
            index_map[label].append(ix)

    for c in range(n_classes):
        to_keep.update(index_map[c][:max(0, n_to_sample[c])])

    return [
        d
        for ix, d in enumerate(data)
        if ix in to_keep
    ]


def scrubbing_evaluation(don_data, max_seq_length, tokenizer, model, args, num_scrubbed):
    print('construct dev tensor')
    dev_data = convert_examples_to_features(
        examples=don_data.dev(),
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
    )
    print('Validation Set Size: ', len(dev_data))
    print('predict dev set')
    
    prediction_data = evaluate(eval_dataset=dev_data, model=model, batch_size=args.batch_size)
    
    print('tuning clf thresholds on dev')
    threshs = tune_threshs(
        probas=prediction_data['pred'],
        truth=prediction_data['truth'],
    )

    # eval
    print("using 'test' for computing test performance")
    print('construct test tensor')
    test_data = convert_examples_to_features(
        examples=don_data.test(),
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
    )
    print('Test Set Size: ', len(test_data))
    print('predict test set')
    
    prediction_data = evaluate(eval_dataset=test_data, model=model, batch_size=args.batch_size)

    # tune thresholds
    print('apply clf thresholds')
    predicted_mat = apply_threshs(
        probas=prediction_data['pred'],
        threshs=threshs,
    )

    print("Result:")
    res = evaluate_multilabels(
        y=multihot_to_label_lists(prediction_data['truth'], don_data.label_map),
        y_preds=multihot_to_label_lists(predicted_mat, don_data.label_map),
        do_print=True,
    )
    filename = "results/NLP/"+"NLP_Scores_class_"+str(args.removal_class)+"_removed_"+str(num_scrubbed)+"_foci_"+args.FOCIType+"_flips_"+str(args.n_tokens_to_replace)+"_epsilon_"+str(args.epsilon)+".csv"
    print("Saving test results to: ", filename)
    df = pd.DataFrame(res)
    if os.path.isfile(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='a', header=True, index=False)


def compute_residual_gradnorm(model, residual_loader, criterion, device):
    total_gradnorm = 0

    for batch in tqdm(residual_loader, desc="Residual Norm Evaluation"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            #'token_type_ids': batch[2],
            'labels': batch[3]
        }

        logits = model(**inputs)
        loss = criterion(logits, inputs['labels'])

        model.zero_grad()
        loss.backward()
        batch_norm = gradNorm(model)

        total_gradnorm += batch_norm

    return total_gradnorm/len(residual_loader)

def main():

    parser = build_arg_parser()
    args = parser.parse_args()
    print(args)
    manual_seed(seed=args.seed)

    if args.mode not in {'test', 'train', 'scrub'}:
        raise ValueError(f"unknown mode {args.mode}, use 'test' or 'train' or 'scrub'")

    if args.subsample_quantile is not None:
        if not (1.0 > args.subsample_quantile > 0.0):
            raise ValueError(
                f"subsampling quantile needs to be None or in (0.0, 1.0),"
                f" given: {args.subsample_quantile}"
            )

    max_seq_length = args.max_seq_len

    don_data = DonData(path=args.data)

    model_name = 'distilbert-base-uncased'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = DistilBertConfig.from_pretrained(model_name, num_labels=len(don_data.all_lbls))
    tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = DistilBertForMultilabelSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )
    model.to(device)

    if args.mode == 'train':

        train_params = {
            'seed': args.seed or 0xDEADBEEF,
            'batch_size': args.batch_size or 8,
            'epochs': args.train_epochs or 1,
            'weight_decay': args.weight_decay or 0.01,
            'learning_rate': args.learning_rate or 5e-5,
            'adam_epsilon': args.adam_epsilon or 1e-8,
            'warmup_steps': args.warmup_steps or 0,
            'max_grad_norm': args.max_grad_norm or 1.0,
        }

        # training
        train_data = don_data.train()
        if args.subsample_quantile is not None:
            print('subsampling training data')
            train_data = subsample(
                data=train_data,
                quantile=args.subsample_quantile,
                n_classes=len(don_data.all_lbls),
            )

        print('construct training data tensor')
        train_data = convert_examples_to_features(
            examples=train_data,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
        )
        print('start training')
        train(
            train_dataset=train_data,
            model=model,
            train_params=train_params,
            class_weights=don_data.class_weights if args.use_class_weights else None,
        )

        torch.save(model, args.model_path)
    

    elif args.mode == "scrub":
        print("************Scrubbing**************")
        print('loading model', args.model_path)
        if torch.cuda.is_available():
            model = torch.load(args.model_path)
        else:
            model = torch.load(args.model_path, map_location='cpu')

        # print(model)
        # training
        print("Loading training data")
        train_data = don_data.train()
        if args.subsample_quantile is not None:
            print('subsampling training data')
            train_data = subsample(
                data=train_data,
                quantile=args.subsample_quantile,
                n_classes=len(don_data.all_lbls),
            )

        print('construct training data tensor')
        train_data = convert_examples_to_features(
            examples=train_data,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
        )

        outString = 'trained_models/'+"Ledgar"+"_"+"distilbert"+"_run_" + str(args.seed) + '_epochs_' + str(args.train_epochs)
    
        tmp = {}
        tmp['class_removed'] = [args.removal_class]
        tmp['n_tokens_to_replace'] = [args.n_tokens_to_replace]
        tmp['train_epochs'] = [args.train_epochs]
        tmp['selectionType'] = [args.selectionType]
        tmp['order'] = [args.order]
        tmp['HessType'] = [args.HessType]
        tmp['approxType'] = [args.approxType]
        tmp['fociType'] = [args.FOCIType]
        tmp['num_perturb'] = [args.n_perturbations]
        tmp['run'] = [args.seed] 
        args.orig_trainset_size = len(train_data)
        tmp['orig_trainset_size'] = [len(train_data)]
        tmp['delta'] = [args.delta]
        tmp['epsilon'] = [args.epsilon]
        tmp['l2_reg'] = [args.l2_reg]

        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

        class_weights=don_data.class_weights if args.use_class_weights else None

        if class_weights is not None:
            class_weights = torch.from_numpy(class_weights).float().to(device)

        criterion = nn.BCEWithLogitsLoss(
                reduction='mean',
                pos_weight=class_weights,
            )

        scrubbed_list = []
        num_scrubbed = 0
        ordering = np.random.permutation(args.orig_trainset_size)
        j = 0
            
        with torch.no_grad():
            print("@@@@@@@@@@@@@@@@@@@@@ For Inital  loaded model @@@@@@@@@@@@@@@@@@@@@")
            scrubbing_evaluation(don_data, max_seq_length, tokenizer, model, args, num_scrubbed)
            print("@@@@@@@@@@@@@@@@@@@@@ For Inital  loaded model @@@@@@@@@@@@@@@@@@@@@")


        # for i in range(args.n_removals):
        while num_scrubbed < args.n_removals:

            # because we reload the model
            no_decay = {'bias', 'LayerNorm.weight'}
            optimizer_grouped_parameters = [
                {
                    'params': [
                        p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    'weight_decay': args.weight_decay,
                },
                {
                    'params': [
                        p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    'weight_decay': 0.0,
                },
            ]
            optim = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
            )

            if args.scrub_batch_size is not None:

                # select batch of samples to scrub

                scrub_list = []
                while len(scrub_list) < args.scrub_batch_size and (num_scrubbed+len(scrub_list)) < args.n_removals:
                    scrubee = ordering[j]

                    num_labels_of_scrubee = torch.count_nonzero(train_data[scrubee][3]).item()
                    if num_labels_of_scrubee>1:
                        scrubee_class = -1
                    else:
                        scrubee_class = torch.nonzero(train_data[scrubee][3]).item()

                    if scrubee_class == args.removal_class:
                        scrub_list.append(scrubee)

                    j += 1

            else:

                # select sample to scrub

                scrubee_class = -1
                while scrubee_class != args.removal_class:
                    scrubee = ordering[j]
                    num_labels_of_scrubee = torch.count_nonzero(train_data[scrubee][3]).item()
                    if num_labels_of_scrubee>1:
                        scrubee_class = -1
                    else:
                        scrubee_class = torch.nonzero(train_data[scrubee][3]).item()

                    j += 1

                scrub_list = [scrubee]
            
            num_scrubbed += len(scrub_list)
            scrub_dataset = Subset(train_data, scrub_list)
            scrubbed_list.extend(scrub_list)

            residual_dataset = SubsetDataWrapper(train_data, include_indices=None, exclude_indices=scrubbed_list)
            residual_loader = torch.utils.data.DataLoader(residual_dataset, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=1)
            
            print('Residual dataset size: ', len(residual_dataset))
            print('Removing: ', num_scrubbed, scrub_list, scrubee_class)

            tmp['scrub_list'] = [scrub_list]
            tmp['n_removals'] = [num_scrubbed]

            foci_val, updatedSD, samplossbefore, samplossafter, gradnormbefore, gradnormafter = inp_perturb(model, scrub_dataset, criterion, args, optim, device, outString=outString, is_nlp=True)

            # reload for deepcopy
            # apply new weights
            print('reloading model for applying new weights')
            model = DistilBertForMultilabelSequenceClassification.from_pretrained(
                model_name,
                config=config,
            )
            model.to(device)
            model.load_state_dict(updatedSD)

            tmp['sample_loss_before'] = [samplossbefore.detach().cpu().item()]
            tmp['sample_loss_after'] = [samplossafter.detach().cpu().item()]
            tmp['sample_gradnorm_before'] = [gradnormbefore]
            tmp['sample_gradnorm_after'] = [gradnormafter]

            resid_gradnorm = compute_residual_gradnorm(model, residual_loader, criterion, device)
            tmp['residual_gradnorm_after'] = [resid_gradnorm]

            tmp['time'] = time.time()

            with torch.no_grad():
                print("@@@@@@@@@@@@@@@@@@@@@ For Reloaded & Updated model @@@@@@@@@@@@@@@@@@@@@")
                scrubbing_evaluation(don_data, max_seq_length, tokenizer, model, args, num_scrubbed)
                print("@@@@@@@@@@@@@@@@@@@@@ For Reloaded & Updated model @@@@@@@@@@@@@@@@@@@@@")

            df = pd.DataFrame(tmp)
            if os.path.isfile(args.outfile):
                df.to_csv(args.outfile, mode='a', header=False, index=False)
            else:
                df.to_csv(args.outfile, mode='a', header=True, index=False)
            
        print("Saving removed data_points")
        np.save("results/NLP/NLP_scrubbed_class_"+str(args.removal_class)+"_foci_"+args.FOCIType+"_flips_"+str(args.n_tokens_to_replace)+"_epsilon_"+str(args.epsilon)+".npy", np.asarray(scrubbed_list))

    else:
        print("Test mode")
        print('loading model', args.model_path)
        if torch.cuda.is_available():
            model = torch.load(args.model_path)
        else:
            model = torch.load(args.model_path, map_location='cpu')

        scrubbing_evaluation(don_data, max_seq_length, tokenizer, model, args, 0)




def build_arg_parser():
    parser = argparse.ArgumentParser()
    # Original arguments from distilbert_baseline file
    parser.add_argument("--data", default=None, type=str, required=True, help="Path to .jsonl file containing dataset")
    parser.add_argument("--mode", default="test", type=str, required=True, help="which mode: 'train' or 'test' or 'scrub'")
    parser.add_argument("--model_path", default='./ledgar_data/model/distilbert.pt', type=str, required=False, help="path to model file, ./ledgar_data/model/distilbert.pt")
    parser.add_argument("--subsample_quantile", default=None, type=float, required=False, help="subsample training data such that every class has at most"
             " as many samples as the quantile provided,"
             " no subsampling if set to None, default None"
    )
    parser.add_argument("--use_class_weights", default=True, type=bool, required=False, help="use balanced class weights for training, default True")
    parser.add_argument("--seed", default=0xDEADBEEF, type=int, required=False, help="seed for random number generation, default 0xDEADBEEF")
    parser.add_argument("--max_seq_len", default=128, type=int, required=False, help="maximum sequence length in transformer, default 128")
    parser.add_argument("--batch_size", default=256, type=int, required=False, help="training batch size, default 8")
    parser.add_argument("--train_epochs", default=1, type=int, required=False, help="number of epochs of training, default 1")
    parser.add_argument("--weight_decay", default=0.01, type=float, required=False, help="AdamW weight decay, default 0.01")
    parser.add_argument("--learning_rate", default=5e-5, type=float, required=False, help="AdamW learning rate, default 5e-5")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, required=False, help="AdamW epsilon, default 1e-8")
    parser.add_argument("--warmup_steps", default=0, type=int, required=False, help="Warmup steps for learning rate schedule, default 0")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, required=False, help="max norm for gradient clipping, default 1.0")


    # Arguments from multi scrub
    parser.add_argument('--dataset', type=str, default='LEDGAR')
    parser.add_argument('--n_removals', type=int, default=100, help='number of samples to scrub')
    parser.add_argument('--removal_class', type=int, default=11, help='id of class to be removed')
    parser.add_argument('--orig_trainset_size', type=int, default=None, help='size of orig training set')
    parser.add_argument('--epsilon', type=float, default=1.0, help='scrubbing rate')
    parser.add_argument('--delta', type=float, default=1.0, help='scrubbing rate')
    parser.add_argument('--l2_reg', type=float, default=0.01, help='weight_decay or l2_reg, used for noisy return and hessian smoothing')
    parser.add_argument('--lr', type=float, default=1.0, help='scrubbing rate')
    parser.add_argument('--scrubType', type=str, default='IP', choices=['IP','HC'])
    parser.add_argument('--HessType', type=str, default='Sekhari', choices=['Sekhari','CR'])
    parser.add_argument('--approxType', type=str, default='FD', choices=['FD','Fisher'])
    parser.add_argument('--n_perturbations', type=int, default=1000)
    parser.add_argument('--order', type=str, default='Hessian', choices=['BP','Hessian'])
    parser.add_argument('--selectionType', type=str, default='FOCI', choices=['Full', 'FOCI', 'Random'])
    parser.add_argument('--FOCIType', type=str, default='full', choices=['full','cheap'])
    parser.add_argument('--cheap_foci_thresh', type=float, default=0.05, help='threshold for codec2 calls in cheap_foci')
    parser.add_argument('--outfile', type=str, default="results/NLP/ledgar_scrub_ablate_results.csv", help='output file name to append to')
    parser.add_argument('--hessian_device', type=str, default='cpu', help='Device for Hessian computation')
    parser.add_argument('--scrub_batch_size', type=int, default=None, help='Batch size for batch scrubbing')
    parser.add_argument('--n_tokens_to_replace', type=int, default=1, help='Number of tokens to replace per sentence to create a perturbation')

    return parser

if __name__ == '__main__':
    main()
