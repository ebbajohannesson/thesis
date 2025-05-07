from datasets import Dataset
from datasets import load_from_disk
import random

# load data
#prestring = "theca"
#prestring = "amazon"
prestring = "msmarco"
qrel_train = load_from_disk(prestring+"_qrel_ds")
query_train = load_from_disk(prestring+"_query_ds_train")
corpus_ds = load_from_disk(prestring+"_corpus_ds")

top_100_labels_train = load_from_disk(prestring+"_top_ds_train")

corpus_id = corpus_ds["product_id"]
p_id_to_index = {p_id: idx for idx,p_id in enumerate(corpus_id)}

query_id = query_train["query_id"]
q_id_to_index = {q_id: idx for idx,q_id in enumerate(query_id)}

# set nbr of positive and negative examples
nbr_pos = 1
nbr_neg = 2

# create "triplets"
m = len(query_train["query"])
queries_texts = query_train["query"]
#query_ids = query_train["query_id"]
queries = []
positives = []
negatives = []
qrel_queries = qrel_train["query_id"]
qrel_pos = qrel_train["product_id"]
negative_examples = []

labels = top_100_labels_train["label"]
product_ids = top_100_labels_train["product_id"]
query_ids = top_100_labels_train["query_id"][::100]
for i in range(m):
    j = 0
    positive_examples = []
    found_id = []
    while len(found_id)<nbr_pos:
        while j<100 and labels[i*100+j] == 0: #hämtar högst rankade relevanta produkten
            j += 1
        if j>=100:
            q_id = query_ids[i]
            idx = qrel_queries.index(q_id)
            while idx is not None and qrel_pos[idx] in found_id:
                try:
                    idx = qrel_queries[idx+1:].index(q_id)+(idx+1)
                except ValueError:
                    idx = None
                    print(f"ingen positiv, q_id: {q_id}")
            if idx is not None:
                positive_examples.append(corpus_ds[p_id_to_index[qrel_pos[idx]]]["combined_text"])#väljer en relevant om ingen fanns i top 100
                found_id.append(qrel_pos[idx])
            else:
                found_id.append(None)
        else:
            positive_examples.append(corpus_ds[p_id_to_index[product_ids[i*100+j]]]["combined_text"])
            found_id.append(product_ids[i*100+j])
        j += 1
    positives.append(positive_examples)

qrel_dict = {}
for row in qrel_train:
    qrel_dict[(row["query_id"], row["product_id"])] = 1

for i in range(m):
    q_id = query_ids[i]
    q_idx = q_id_to_index[q_id]
    queries.append(queries_texts[q_idx])
    j = 99
    negative_examples = []
    found_idx = []
    """
    while len(found_idx)<nbr_neg:
        while labels[i*100+j] == 1: #hämtar högst rankade relevanta produkten
            j -= 1
        if j<0:
            print("inga negativa kvar!")
            print(f"q_id: {q_id}")
            idx = random.randint(0,len(corpus_ds))
            while (q_id, corpus_id[idx]) in qrel_dict or idx in found_idx: #väljer random icke-relevant produkt
                idx = random.randint(0,len(corpus_ds))
            negative_examples.append(corpus_ds[idx]["combined_text"])
            found_idx += [idx]
        else:
            negative_examples.append(corpus_ds[p_id_to_index[product_ids[i*100+j]]]["combined_text"])
            found_idx += [j]
        j-=1
    negatives.append(negative_examples)
    """
    while len(found_idx)<nbr_neg:
        idx = random.randint(0,len(corpus_ds))
        while (q_id, corpus_id[idx]) in qrel_dict or idx in found_idx:
            idx = random.randint(0,len(corpus_ds))
        negative_examples.append(corpus_ds[idx]["combined_text"])
        found_idx += [idx]
    negatives.append(negative_examples)

train_dict = {"query": queries, "positive": positives, "negative": negatives}
train_dataset_triplet = Dataset.from_dict(train_dict).shuffle(seed=42)
#train_dataset_triplet = Dataset.from_dict(train_dataset_triplet[:len(train_dataset_triplet)//5])

# combine queries with examples
pairs = []
pairs_pos = []
pairs_neg = []
labels = []
for triplet in train_dataset_triplet: # kan göra direkt från dict också
    for pos in triplet["positive"]:
        pairs.append([triplet["query"], pos])
        labels.append(1)
        #for _ in range(nbr_neg): #for marginrankingloss, don´t have to comment out
        #    pairs_pos.append([triplet["query"], pos])
    for neg in triplet["negative"]:
        pairs.append([triplet["query"], neg])
        labels.append(0)
        #pairs_neg.append([triplet["query"], neg]) #for marginrankingloss, don´t have to comment out

train_dict = {"data": pairs, "label": labels}
#train_dict = {"data_pos": pairs_pos, "data_neg": pairs_neg}
train_dataset = Dataset.from_dict(train_dict)

train_dataset.save_to_disk(prestring+"_TRAIN12_random")