import pickle
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import os
torch.cuda.set_device(1)

"""
model_sbert_TH_78000k_c_emb.pkl                model_sbert_TH_78000k_physbert_c_emb.pkl
model_sbert_TH_78000k_gpu1_c_emb.pkl           model_sbert_TH_78000k_physbert_q_emb.pkl
model_sbert_TH_78000k_gpu1_q_emb.pkl           model_sbert_TH_78000k_q_emb.pkl
model_sbert_TH_78000k_materialsbert_c_emb.pkl  model_sbert_TH_78000k_scinewsbert_c_emb.pkl.filepart
model_sbert_TH_78000k_materialsbert_q_emb.pkl  model_sbert_TH_78000k_scinewsbert_q_emb.pkl
"""



directory = "numpy_embedding_trian_base"

os.makedirs(directory, exist_ok=True)

#data_path = "./data/protective_papers_dataset_78000_clean_test.json"
data_path = "./data/protective_papers_dataset_78000_clean_train.json"

models_path = ['model_sbert_TH_78000k_',
'model_sbert_TH_78000k_gpu1_',
'model_sbert_TH_78000k_physbert_',
'model_sbert_TH_78000k_scinewsbert_',
'model_sbert_TH_78000k_materialsbert_',
'matscibert_base_',
'all-mpnet-base-v2_base_',
'MaterialsBERT_base_',
'physbert_cased_base_',
'SciNewsBERT_'
]
#path_to_dir = './embedding_test/'
#path_to_dir = './embedding_train/'
path_to_dir = './embedding_train_base/'
id_help = 0

############################################################

from typing import List, Dict
import json

# =========================
# DATA LOADING
# =========================

def load_documents_from_json(path: str):
    print(f"Loading data from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} papers")
    return documents


def chunk_text(text, chunk_size=300):
    words = text.split()
    return [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def build_documents(papers):
    """
    Tworzy dataset (abstract, chunk)
    """
    documents = []
    id_help = 0
    for p in papers:
        abstract = p.get("abstract", "").strip()
        body = p.get("body", "").strip()

        if not abstract or not body:
            continue

        chunks = chunk_text(body)

        for c in chunks:
            documents.append({
                "abstract_id": id_help,
                "abstract": abstract,
                "chunk": c
            })
        id_help = id_help + 1

    print(f"Generated {len(documents)} (abstract, chunk) pairs")
    return documents


def prepare_data(documents: List[Dict]):
    abstracts_ids = [d["abstract_id"] for d in documents]
    queries = [d["abstract"] for d in documents]
    corpus = [d["chunk"] for d in documents]
    return abstracts_ids, queries, corpus

############################################################

papers = load_documents_from_json(data_path)
documents = build_documents(papers)
abstracts_ids, queries, corpus = prepare_data(documents)

for id_help in range(5,10):#len(models_path)):
    cc = None
    qq = None
    with open(path_to_dir + models_path[id_help] + "c_emb.pkl", "rb") as f:
        cc = pickle.load(f)
        cc = cc.to(torch.device("cuda:1"))
    with open(path_to_dir + models_path[id_help] + "q_emb.pkl", "rb") as f:
        qq = pickle.load(f)
        qq = qq.to(torch.device("cuda:1"))



    i = 0
    hits_list = []
    top_k = 0

    for i in tqdm(range(cc.shape[0])):
        top1 = []

        scores = util.cos_sim(qq[i], cc)[0]
        ranked = torch.argsort(scores, descending=True)

        smallest_id_ranked = abstracts_ids[ranked[0].item()]
        smallest_id_i = abstracts_ids[i]

        if smallest_id_ranked == smallest_id_i:
            hits_list.append(True)
        else:
            hits_list.append(False)

    with open(directory + "/" + models_path[id_help] + ".pkl", "wb") as f:
        pickle.dump(hits_list, f)

    print(str(sum(hits_list) / len(hits_list)))

