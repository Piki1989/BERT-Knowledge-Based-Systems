####################################
#
####################################
print("# BERT PROTECTIVE #")

import os
import random
import json
import numpy as np
import torch
from tqdm.auto import tqdm

import os
from time import gmtime, strftime
from pathlib import Path
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["PYTORCH_ENABLE_SDPA"] = "0"
#os.environ["PYTORCH_SDPA_ENABLE_FLASH"] = "0"
#os.environ["PYTORCH_SDPA_ENABLE_MEM_EFFICIENT"] = "0"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "globuslabs/ScholarBERT-XL"
#DATA_PATH =  "/home/shared_bert/protection/protective_papers_dataset_78000_clean.json"#"../data/Output.json"
#DATA_PATH =  "./data/Output.json"
DATA_PATH =  "./data/protective_papers_dataset_78000_clean.json"

split_idx_file = "./split_idx_78000.csv"

MAX_LENGTH = 512
CHUNK_SIZE = 300

MLM_EPOCHS = 1
SBERT_EPOCHS = 2
LR = 2e-5

MLM_DIR = "./model_dapt_TH_78000kt_scholarbertxl2"
SBERT_DIR = "./model_sbert_TH_78000k_scholarbertxl2"
#index_file_res = "armor_index_TH.faiss_test"

#DEVICE = "cuda" if torch.cuda.is_available() else "mps"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


####################################

def create_folder(dir_name = ""):
    dir_name_help = dir_name + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "/"
    Path(dir_name_help).mkdir(parents=True, exist_ok=True)
    return dir_name_help

def reaturn_folder(dir_name = "", sub_dir_name = ""):
    file_name_help = dir_name + sub_dir_name + strftime("%Y-%m-%d %H:%M:%S", gmtime())
    return file_name_help

####################################

print("# Loading data #")

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

with open(DATA_PATH) as f:
    papers = json.load(f)

documents = []
keyword_pool = []
print("All papers = " + str(len(papers)))

random.seed(42)
#random.shuffle(papers)
split_idx = np.random.choice(np.arange(len(papers), dtype=int), size=int(0.9 * len(papers)), replace=False).astype(int, copy=False)
print(split_idx.shape)
print(split_idx[0:10])
np.savetxt(split_idx_file, split_idx, delimiter=",")
papers_ = [papers[i] for i in split_idx]
papers = papers_
print("Train papers = " + str(len(papers)))
"""
print(papers[0]["title"])
print(papers_[0]["title"])
print(papers[split_idx[0]]["title"])
print(papers_[1]["title"])
print(papers[split_idx[1
    ]]["title"])

for a in range(10):
    print(a)
print("Train papers = " + str(len(papers)))
a = 0 / 0
"""
#for p in papers:
#for p in papers[0:int(len(papers) * 0.01)]:
for p_id in tqdm(range(len(papers))):
    p = papers[p_id]
    abstract = p.get("abstract", "")
    body = p.get("body", "")
    keywords = p.get("keywords", [])

    keyword_pool.extend(keywords)

    chunks = chunk_text(body)

    for c in chunks:
        documents.append({
            "abstract": abstract,
            "chunk": c,
            "keywords": keywords
        })

len("data size = " + str(documents))

####################################

from torch.utils.data import DataLoader
from datasets import Dataset

from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling
)

# -------------------------
# 1. Przygotowanie danych
# -------------------------
texts = [
    d["abstract"] + " " + d["chunk"]
    for d in documents
]

dataset = Dataset.from_dict({"text": texts})

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="longest",
        max_length=MAX_LENGTH
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# -------------------------
# 2. Model + collator
# -------------------------

print("# Masaked language model pretraining #")

model = BertForMaskedLM.from_pretrained(MODEL_NAME)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# -------------------------
# 3. DataLoader
# -------------------------
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=data_collator
)

# -------------------------
# 4. Optymalizator
# -------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------
# 5. Pętla treningowa PyTorch
# -------------------------
model.train()

global_step = 0
max_steps = 5000

for epoch in range(MLM_EPOCHS):
    for batch in dataloader:

        # zatrzymaj gdy przekroczymy limit
        if global_step >= max_steps:
            break

        # przeniesienie batcha na GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        # forward
        outputs = model(**batch)
        loss = outputs.loss

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1

        # logging
        if global_step % 200 == 0:
            print(f"Step {global_step} | Loss: {loss.item():.4f}")

    if global_step >= max_steps:
        break

# -------------------------
# 6. Zapis modelu
# -------------------------
model.save_pretrained(MLM_DIR)
tokenizer.save_pretrained(MLM_DIR)

print("# Pretraining completed #")

####################################

print("# Generating dataset #")

from sentence_transformers import InputExample

train_examples = []

for d in documents:

    # abstract ↔ chunk
    train_examples.append(
        InputExample(texts=[d["abstract"], d["chunk"]])
    )

    # chunk ↔ keyword
    for kw in d["keywords"]:
        train_examples.append(
            InputExample(texts=[d["chunk"], kw])
        )

len("# Dataset size = " + str(train_examples))

####################################

print("# Model training #")

import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

# Load model
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(MLM_DIR)
model.train()

# Dataloader with correct collate
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=4,
    collate_fn=model.smart_batching_collate
)

# Loss
loss_fn = losses.MultipleNegativesRankingLoss(model)

num_epochs = SBERT_EPOCHS
num_training_steps = len(train_dataloader) * num_epochs
warmup_steps = int(len(train_dataloader) * 0.1)

optimizer = AdamW(model.parameters(), lr=2e-5)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)

scaler = GradScaler()

########################

my_save_dir_name = "HP_"
sub_dir_name = ""
save_folder_name = create_folder(dir_name = my_save_dir_name)
save_every = int(len(train_dataloader) / 5)
print("save every = " + str(save_every))
########################

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    step = 0
    loop = tqdm(train_dataloader) # We use this to display a progress bar
    for batch in loop:
        features, labels = batch
        features_on_gpu = [{k: v.to(device, non_blocking=True) for k, v in f.items()} for f in features]
        labels_on_gpu = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        #print("####################################################")
        #print(features)
        #print("####################################################")
        with autocast():
            loss = loss_fn(features_on_gpu, labels_on_gpu)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if (step + 1) % 10 == 0:  
            loop.set_description("Epoch: {}".format(epoch)) # Display epoch number
            loss_help = loss.item()
            loop.set_postfix(loss=loss_help) # Show loss in the progress bar
            with open("bert_train_res.txt", "a") as myfile:
                myfile.write(str(step) + ", " + str(loss_help) + "\n")
        step += 1

        if step % save_every == 0:
            save_folder_name_subfolder = reaturn_folder(save_folder_name, sub_dir_name)
            model.save(save_folder_name_subfolder + "_model_epoch" + str(epoch))
        
    print("Epoch loss:", loss.item())


# Save model
model.save(SBERT_DIR)
print("# Saved model to:" + SBERT_DIR + " #")

####################################

#faiss.write_index(index, index_file_res)
