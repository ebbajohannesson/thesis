import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import Dataset
import numpy as np
from sklearn import svm
import faiss

def crossencoder_reranker(model: nn.Module, tokenizer, top_k: Dataset, device): #top_k får innehålla texter också (och embeddings?)

    # t_start
    query_product_text = []
    query_text = top_k["query_text"][0] #top_k innehåller bara för en query
    product_texts = top_k["product_text"]
    for p_text in product_texts:
        query_product_text.append([query_text, p_text])

    tokens = tokenizer(query_product_text, padding=True, truncation=True, return_tensors="pt")
    """
    tokenized_dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"])
    dataloader = DataLoader(tokenized_dataset, batch_size=50, shuffle=False) #borde typ inte ens behöva batches för 100 produkter?
    """
    tokens.to(device)
    model.to(device)

    model.eval()
    #logits = []
    with torch.no_grad():
        """for input_ids, attention_mask in dataloader: #kan skippa mkt om inte dataloader/batches nehövs
                tokens = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)} #får inte till på nåt annat sätt
                """
        logits = model(**tokens)
        #logits.append(outputs)
        #logits = torch.cat(logits)
        soft = nn.Softmax(dim=-1)
        scores = soft(logits)[:,1]
    # t_stop
    # time = t_stop-t_start
    return scores #, time

def svm_reranker(top_k: Dataset, corpus_embeddings: np.array): #top_k får innehålla embeddings
    # t_start
    query_embedding = top_k["query_embedding"][0] #top_k innehåller bara för en query
    product_embeddings = top_k["product_embedding"]

    dimension = query_embedding.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(corpus_embeddings)
    index.add(corpus_embeddings)
    faiss.normalize_L2(query_embedding)
    k = 100
    _, I = index.search(-1*query_embedding, k)

    negative_embeddings = corpus_embeddings[I] # tror det här ska funka med indexen

    x_train = np.concatenate([query_embedding[None,...], negative_embeddings])
    x_rank = np.concatenate([query_embedding[None,...], product_embeddings])
    y = np.zeros(len(negative_embeddings)+1)
    y[0] = 1

    clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=1, dual='auto', loss="hinge")
    clf.fit(x_train,y)

    scores = clf.decision_function(x_rank)[1:]
    # t_stop
    # time = t_stop-t_start
    return scores #, time

def biencoder_reranker():

    return



# ladda top-k dataset (en query kommer k ggr med annan produkt/dokument)
# med query-texter, produkt/dokument-texter, query-embeddings, produkt/dokument-embeddings, label för paret

# loop där k rader i datasetet (en query) väljs i taget och skickas till en reranker
# tider och scores sparas emellan

