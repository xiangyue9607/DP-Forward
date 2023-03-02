import random
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification,BertTokenizer,BertForMaskedLM
from tqdm import tqdm
import torch
from dp_noise import matrix_gaussian_noise
import argparse
import os
from torch.optim.lr_scheduler import StepLR
from datasets import load_dataset



model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model_dp = BertForSequenceClassification.from_pretrained("sst2_eps8_embedding")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model_dp.to(device)
model_dp.eval()
model.eval()


def embedding_inversion_basic(test_docs,epsilon=30,delta=1e-5,layer=0,iteration=2000,lr=1e-4,T = 0.05):

    test_dataloader = DataLoader(test_docs, sampler=SequentialSampler(test_docs), batch_size=batch_size)
    for jj, cur_batch in enumerate(tqdm(test_dataloader)):
        cur_max_length = cur_batch[1].max().item()
        cur_batch = cur_batch[0]
        cur_batch = tokenizer.batch_encode_plus(cur_batch, max_length=cur_max_length, truncation=True,
                                                padding='max_length')
        input_ids = torch.tensor(cur_batch["input_ids"],dtype=torch.long, device=device)
        attention_mask = torch.tensor(cur_batch['attention_mask'],dtype=torch.long, device=device)
        noise_b = matrix_gaussian_noise(epsilon=epsilon, delta=delta, sensitivity=1.0)
        noise_b = 0
        if layer==0:
            with torch.no_grad():
                embedding_matrix_private = model_dp.bert.embeddings.word_embeddings(input_ids)
                noise_embeds = noise_b * torch.normal(mean=0, std=1, size=embedding_matrix_private.shape).to(device)
                embedding_matrix = noise_embeds + embedding_matrix_private

            z = torch.rand((embedding_matrix.shape[0], embedding_matrix.shape[1], model.bert.embeddings.word_embeddings.weight.shape[0]), requires_grad=True, device=device)
            torch.nn.init.xavier_uniform(z)

            mse_loss = torch.nn.MSELoss()
            optimizer = torch.optim.Adam([z], lr=lr)

            for _ in range(iteration):
                v = torch.softmax(z / T, dim=-1)

                approx_embedding_matrix = torch.matmul(v, model.bert.embeddings.word_embeddings.weight.data)

                loss = mse_loss(approx_embedding_matrix, embedding_matrix)
                loss.backward()
                optimizer.step()
                z.grad.zero_()
                # print(loss)


            z = torch.argmax(z, dim=-1)
            z = z * attention_mask

            correct = torch.sum(input_ids == z) - torch.sum(attention_mask == 0)
            total = torch.sum(attention_mask)
            acc = 1.0 * correct / total
            print(acc.item())

    return correct.item(),total.item(),noise_b

def embedding_inversion_similarity(batch,epsilon=30,delta=1e-5,layer=0):
    input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
    attention_mask = torch.tensor(batch['attention_mask'], dtype=torch.long, device=device)
    noise_b = matrix_gaussian_noise(epsilon=epsilon, delta=delta, sensitivity=1.0)
    noise_b = 0
    if layer==0:
        with torch.no_grad():
            embedding_matrix_private = model_dp.bert.embeddings.word_embeddings(input_ids)
            noise_embeds = noise_b * torch.normal(mean=0, std=1, size=embedding_matrix_private.shape).to(device)
            embedding_matrix = noise_embeds + embedding_matrix_private

    # z = torch.matmul(embedding_matrix, model.bert.embeddings.word_embeddings.weight.data.T)
    z = torch.cdist(embedding_matrix, model.bert.embeddings.word_embeddings.weight.data)
    z = torch.argmin(z, dim=-1)
    z = z * attention_mask

    correct = torch.sum(input_ids == z) - torch.sum(attention_mask == 0)
    total = torch.sum(attention_mask)
    acc = 1.0 * correct / total
    # print(acc.item())
    return correct.item(), total.item(), noise_b



def output_embedding_inversion(train_docs,test_docs,epsilon=30,delta=1e-5,lr=1e-2,epochs=1,iteration=5000, T=0.05, layer=12):
    hidden_size = model_dp.config.hidden_size
    mapping_matrix = torch.rand((hidden_size,hidden_size), requires_grad=True, device=device)
    torch.nn.init.xavier_uniform(mapping_matrix)

    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([mapping_matrix], lr=lr)
    schedular = StepLR(optimizer, step_size=20, gamma=0.9)
    train_dataloader = DataLoader(train_docs, sampler=RandomSampler(train_docs), batch_size=batch_size)

    for _ in range(epochs):
        for jj,cur_batch in enumerate(tqdm(train_dataloader)):
            cur_max_length = cur_batch[1].max().item()
            cur_batch = cur_batch[0]
            cur_batch = tokenizer.batch_encode_plus(cur_batch, max_length=cur_max_length, truncation=True,
                                                    padding='max_length')
            input_ids = torch.tensor(cur_batch["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(cur_batch['attention_mask'], dtype=torch.long, device=device)

            _, output_embedding = model_dp(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            output_embedding = (output_embedding[layer]*attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            mapping_embedding = torch.matmul(output_embedding, mapping_matrix)

            input_embedding = model_dp.bert.embeddings.word_embeddings(input_ids)
            input_embedding = input_embedding.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

            loss = mse_loss(input_embedding, mapping_embedding)
            loss.backward()
            optimizer.step()
            schedular.step()
            mapping_matrix.grad.zero_()

            print(loss.item(),schedular.get_lr())

    torch.save(mapping_matrix,open("mapping_matrix.pt",'wb'))

    mapping_matrix = torch.load(open("mapping_matrix.pt",'rb'))
    #######testing 2: equation 7
    test_dataloader = DataLoader(test_docs, sampler=SequentialSampler(test_docs), batch_size=batch_size)

    correct = 0
    total1 = 0
    total2 = 0
    for jj, cur_batch in enumerate(tqdm(test_dataloader)):
        cur_max_length = cur_batch[1].max().item()
        cur_batch = cur_batch[0]
        cur_batch = tokenizer.batch_encode_plus(cur_batch, max_length=cur_max_length, truncation=True,
                                                padding='max_length')
        input_ids = torch.tensor(cur_batch["input_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(cur_batch['attention_mask'], dtype=torch.long, device=device)


        with torch.no_grad():
            input_embedding = model_dp.bert.embeddings.word_embeddings(input_ids)
            input_embedding = input_embedding.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

            # _, output_embedding = model_dp.bert(input_ids=input_ids, attention_mask=attention_mask)
            _, output_embedding = model_dp(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            output_embedding = (output_embedding[layer]*attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            mapping_embedding = torch.matmul(output_embedding, mapping_matrix)

        z = torch.rand(
            (input_embedding.shape[0], model_dp.bert.embeddings.word_embeddings.weight.shape[0]),
            requires_grad=True, device=device)

        torch.nn.init.xavier_uniform(z)
        z.data = torch.clamp(z.data, min=0)

        mse_loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([z], lr=lr)
        for _ in range(iteration):
            approx_embedding = torch.matmul(z, model_dp.bert.embeddings.word_embeddings.weight)
            loss = mse_loss(approx_embedding, mapping_embedding)
                   # + 0.1 * torch.norm(z, 1)
            loss.backward()
            optimizer.step()
            # schedular.step()
            z.data = torch.clamp(z.data, min=0)
            z.grad.zero_()
            # print(loss)

        # threshold = 2e-4
        scores = z.argsort(descending=True)
        for i in range(len(z)):
            prediction_num = 128
            prediction = scores[i, :prediction_num].tolist()
            # prediction = torch.where(z[i]>threshold)[0].tolist()
            ground_truth = input_ids[i].tolist()[:attention_mask[i].sum()]
            correct += len(set(prediction).intersection(set(ground_truth)))
            total1 += len(ground_truth)
            total2 += len(prediction)
        print(correct/total1, correct/total2)



    return correct, total1

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument("--epsilon", type=float, default=1.0, help="dp noise parameter: epsilon")


    args = parser.parse_args()

    train_data_file = os.path.join(args.data_dir,"train.tsv")
    test_data_file = os.path.join(args.data_dir,"dev.tsv")

    max_length = 128
    batch_size = 32
    test_docs = []
    train_docs = []
    if args.data_dir == 'imdb':
        raw_datasets = load_dataset('imdb')
        for i, ex in enumerate(tqdm(raw_datasets['test'])):
            text = ex['text']
            length = len(tokenizer.tokenize(text))
            test_docs.append((text, length))
    else:
        num_lines = sum(1 for _ in open(train_data_file))
        with open(train_data_file) as rf:
            next(rf)
            for line in tqdm(rf, total=num_lines - 1):
                content = line.strip().split("\t")
                text = content[0]
                length = 128
                train_docs.append((text, length))

        num_lines = sum(1 for _ in open(test_data_file))
        with open(test_data_file) as rf:
            next(rf)
            for line in tqdm(rf, total=num_lines - 1):
                content = line.strip().split("\t")
                text = content[0]
                length = len(tokenizer.tokenize(text))
                test_docs.append((text,length))

    # test_docs = random.choices(test_docs,k=1000)
    test_docs = sorted(test_docs,key=lambda x: -x[1])

    results = []
    epsilon = args.epsilon

    output_embedding_inversion(train_docs,test_docs,epochs=1, epsilon=epsilon)


