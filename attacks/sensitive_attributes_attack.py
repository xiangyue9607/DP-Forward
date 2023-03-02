import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import BertConfig
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from modeling_bert_sentence_dp_proj import BertForSequenceClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttackModel(nn.Module):
    def __init__(self, bertmodel, num_labels):
        super().__init__()
        bertmodel.config.output_hidden_states = True
        self.bertmodel = bertmodel
        for p in self.bertmodel.parameters():
            p.requires_grad = False
        self.noise_para = self.bertmodel.config.noise_para
        hidden_size = bertmodel.config.hidden_size
        hidden_dropout_prob = bertmodel.config.hidden_dropout_prob
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(hidden_dropout_prob),
        )
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, inputs):
        hidden_states = self.bertmodel(**inputs)[1]
        hidden_states = self.mlp(hidden_states)
        logits = self.classifier(hidden_states)
        return logits


def load_and_process_datasets(tokenizer, labels, max_seq_length=64, overwrite_cache=False):
    def _process_function(data):

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        all_labels = []
        for text, label in tqdm(data):
            result = tokenizer(text, padding="max_length", max_length=max_seq_length, truncation=True)
            all_input_ids.append(result['input_ids'])
            all_attention_mask.append(result['attention_mask'])
            all_token_type_ids.append(result['token_type_ids'])
            all_labels.append(label2ids[label])

        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
        all_labels = torch.tensor(all_labels, dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

        return dataset

    cached_features_file = "cached_sensitive_attributes_attack_dataset_labels{}_maxlen{}".format(
        len(labels),
        str(max_seq_length),
    )

    if os.path.exists(cached_features_file) and not overwrite_cache:
        print("Loading data from cached file ", cached_features_file)
        train_dataset, dev_dataset, test_dataset = torch.load(cached_features_file)
    else:
        print("Process data...")
        train_data = []
        test_data = []
        label2ids = {label: i for i, label in enumerate(labels)}

        raw_datasets = load_dataset('imdb')
        for i, ex in enumerate(tqdm(raw_datasets['train'])):
            content = ex['text']
            for type in labels:
                if type in content:
                    train_data.append((content, type))
                    break
        for i, ex in enumerate(tqdm(raw_datasets['test'])):
            content = ex['text']
            for type in labels:
                if type in content:
                    train_data.append((content, type))
                    break

        raw_datasets = load_dataset("glue", "sst2")
        for i, ex in enumerate(tqdm(raw_datasets['train'])):
            content = ex['sentence']
            for type in labels:
                if type in content:
                    test_data.append((content, type))
                    break
        for i, ex in enumerate(tqdm(raw_datasets['validation'])):
            content = ex['sentence']
            for type in labels:
                if type in content:
                    test_data.append((content, type))
                    break

        train_data, dev_data = train_test_split(train_data, test_size=int(0.1 * len(train_data)))

        train_dataset = _process_function(train_data)
        dev_dataset = _process_function(dev_data)
        test_dataset = _process_function(test_data)

        torch.save((train_dataset, dev_dataset, test_dataset), cached_features_file)

    return train_dataset, dev_dataset, test_dataset


def train(model, optimizer, scheduler, train_loader, dev_loader, test_loader, labels, epoch=1):
    model.train()
    model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    losses = []
    best_acc = 0
    for _ in range(epoch):
        for _batch_idx, inputs in enumerate(tqdm(train_loader)):
            inputs = tuple(t.to(device) for t in inputs)
            batch = {
                "input_ids": inputs[0],
                "attention_mask": inputs[1],
                "token_type_ids": inputs[2]
            }
            label = inputs[3]

            output = model(batch)
            loss = criterion(output, label)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if (_batch_idx + 1) % 100 == 0 or (_batch_idx + 1) == len(train_loader):
                print(
                    f"Lr: {scheduler.get_lr()[0]} \t"
                    f"Loss: {np.mean(losses):.6f} "
                )

        dev_acc = test(model, dev_loader, labels, split='dev')

        if dev_acc > best_acc:
            best_acc = dev_acc
            test(model, test_loader, labels, split='test')


def test(model, test_loader, labels, split='dev'):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    correct_details = torch.zeros(len(labels), device=device)
    total_details = torch.zeros(len(labels), device=device)
    with torch.no_grad():
        for inputs in tqdm(test_loader):
            inputs = tuple(t.to(device) for t in inputs)
            batch = {
                "input_ids": inputs[0],
                "attention_mask": inputs[1],
                "token_type_ids": inputs[2]
            }
            label = inputs[3]
            output = model(batch)
            test_loss += criterion(output, label).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

            correct_details += torch.sum(
                nn.functional.one_hot(label[torch.where(pred.view_as(label) == label)], num_classes=len(labels)), dim=0)
            total_details += torch.sum(nn.functional.one_hot(label, num_classes=len(labels)), dim=0)
    test_loss /= len(test_loader.dataset)

    accuracy_details = correct_details / total_details
    print(
        "\n{} set: Average loss: {:.4f}, Overall Accuracy: {}/{} ({:.2f}%)\n".format(
            split,
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    print("Accuracy for each class: ")
    for ii in range(len(labels)):
        print(labels[ii], accuracy_details[ii].item())

    return correct / len(test_loader.dataset)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model ",
    )
    parser.add_argument("--seed", type=int, default=32, help="random seed for initialization")

    args = parser.parse_args()

    set_seed(seed=42)

    labels = ['action', 'comedy', 'drama', 'horror']

    train_batch_size = 64
    test_batch_size = 256
    learning_rate = 2e-5
    epoch = 50
    print(train_batch_size, learning_rate, epoch)

    config = BertConfig.from_pretrained(args.model_name_or_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertForSequenceClassificationLDP.from_pretrained(args.model_name_or_path, config=config)

    train_dataset, dev_dataset, test_dataset = load_and_process_datasets(tokenizer, labels, overwrite_cache=False)

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=train_batch_size)
    dev_loader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=test_batch_size)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=test_batch_size)

    model = AttackModel(bertmodel=model, num_labels=len(labels))
    model = model.to(device)
    optimizer_grouped_parameters = [{
        "params": [p for n, p in model.named_parameters() if 'bert' not in n],
        "weight_decay": 0,
    }]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=epoch * len(train_loader)
    )

    train(model, optimizer, scheduler, train_loader, dev_loader, test_loader, labels, epoch)


if __name__ == "__main__":
    main()