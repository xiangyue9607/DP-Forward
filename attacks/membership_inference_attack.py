import argparse
import logging
import os
import random

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertTokenizer,
    InputExample,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_processors as processors

from mia_utils import eda
from modeling_bert_sentence_dp_proj import BertForSequenceClassification
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_and_cache_examples(args, task, tokenizer, in_training=True, sample_num=-1):
    task_to_keys = {
        "sst2": ("sentence", None),
        "qnli": ("question", "sentence"),
        "imdb": ("text", None)
    }
    sentence1_key, sentence2_key = task_to_keys[task]

    # Load data features from cache or dataset file
    cached_features_file = "cached_MIA_{}_{}_{}".format(
        "in_training" if in_training else "not_in_training",
        str(args.max_seq_length),
        str(task),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info(f"Creating features for {task} dataset")

        if task == 'imdb':
            raw_datasets = load_dataset(task)
            split = 'train' if in_training else 'test'
        else:
            raw_datasets = load_dataset("glue", task)
            split = 'train' if in_training else 'validation'

        examples = []
        for i, ex in enumerate(raw_datasets[split]):
            guid = "%s-%s" % ("in_training" if in_training else "not_in_training", i)
            text_a = ex[sentence1_key] if sentence1_key else None
            text_b = ex[sentence2_key] if sentence2_key else None

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=str(ex['label'])))
        if sample_num > 0 and in_training:
            examples = random.sample(examples, sample_num)

        if not in_training:
            for ex in tqdm(examples, desc="randomly modify examples not in training"):
                try:
                    ex.text_a = random.choice(eda(ex.text_a))
                    ex.text_b = random.choice(eda(ex.text_b)) if ex.text_b else ex.text_b
                except:
                    pass

        label_list = ['0', '1']
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            label_list=label_list,
            output_mode='classification',
        )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def evaluate(args, model, tokenizer, in_training=True, sample_num=-1):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task = args.task_name
    eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, in_training=in_training, sample_num=sample_num)

    args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(eval_task))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    losses = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
            losses.append(tmp_eval_loss.cpu().numpy())
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    # do not assume task label available
    if args.attack_method == "confidence":
        preds_prob = np.max(softmax(preds, axis=1), axis=1)

    # assume task label available
    elif args.attack_method == "entropy":
        preds_prob = np.sum(np.multiply(softmax(preds, axis=1),
                                        np.concatenate((1 - out_label_ids.reshape(-1, 1), out_label_ids.reshape(-1, 1)),
                                                       axis=1)), axis=1)

    else:
        raise NotImplementedError(f"Attack method should be either 'confidence' or 'entropy' ")

    preds = np.argmax(preds, axis=1)
    result = compute_metrics("sst-2", preds, out_label_ids)

    return result, preds_prob, out_label_ids


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--attack_method",
        default="entropy",
        type=str,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--per_device_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "device: %s",
        args.device,
    )
    set_seed(args)

    if 'sst' in args.task_name.lower():
        args.task_name = 'sst2'
    elif 'qqp' in args.task_name.lower():
        args.task_name = 'qqp'

    config = BertConfig.from_pretrained(args.model_name_or_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    model.to(args.device)

    max_accs = []
    for i in range(5):
        logger.info(f"Running {i + 1}/5 round...")
        test_result, preds_prob_not_in_training, out_label_ids_not_in_training = evaluate(args, model, tokenizer,
                                                                                          in_training=False)
        train_result, preds_prob_in_training, out_label_ids_in_training = evaluate(args, model, tokenizer,
                                                                                   in_training=True, sample_num=len(
                preds_prob_not_in_training))
        logger.info(test_result)
        accs = []
        thresholds = []
        predictions = [(pred, 1) for pred in preds_prob_in_training] + [(pred, 0) for pred in
                                                                        preds_prob_not_in_training]
        random.shuffle(predictions)

        dev, test = train_test_split(predictions, test_size=0.5)

        probs = np.array([d[0] for d in dev])
        labels = np.array([d[1] for d in dev])

        sample_list = np.random.choice([a for a in range(len(preds_prob_in_training))],
                                       len(preds_prob_in_training) // 2, replace=False)
        validation_set = set(list(sample_list))
        test_set = set([a for a in range(len(preds_prob_in_training))]).difference(validation_set)

        preds_prob_in_training_testset = preds_prob_in_training[list(test_set)]
        preds_prob_not_in_training_testset = preds_prob_not_in_training[list(test_set)]

        preds_prob_in_training = preds_prob_in_training[list(validation_set)]
        preds_prob_not_in_training = preds_prob_not_in_training[list(validation_set)]

        for threshold in np.sort(np.append(preds_prob_in_training, preds_prob_not_in_training)):
            acc = (np.sum([preds_prob_in_training > threshold]) + np.sum([preds_prob_not_in_training <= threshold])) / (
                    len(preds_prob_in_training) + len(preds_prob_not_in_training))

            thresholds.append(threshold)
            accs.append(acc)

        best_threshold = probs[np.argmax(accs)]

        probs = np.array([d[0] for d in test])
        labels = np.array([d[1] for d in test])

        pred = 1 * (probs >= best_threshold)
        acc = accuracy_score(labels, pred)

        max_accs.append(acc)
        print(max_accs)

    max_accs = np.array(max_accs)
    mean = np.mean(max_accs)
    std = np.std(max_accs)
    logger.info(f"mean: {mean}, std: {std}")


if __name__ == "__main__":
    main()