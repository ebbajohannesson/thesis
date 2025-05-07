import torch
import numpy as np
from datasets import Dataset
from datasets import load_from_disk
import pandas as pd
import faiss 
from sklearn.metrics import ndcg_score
import time
from math import ceil

# corpus_ds: "product_id", "combined_text"
# query_ds:  "query", "query_id"
# qrel_ds: "query_id", "product_id"
# top_100_ds: "label", "score", "query_id", "product_id", "product_embedding", "corpus_embedding"

# TODO: lägg in för cpu i crossencodern


RETRIEVAL = False
RERANK = True
device = "cuda"

# need to select what dataset,this determines the corpus_ds, qrel_ds, query_ds and top_100 retrieval dataset we use
dataset = "AMAZON"
#dataset = "THECA"
#dataset = "MSMARCO"

#reranker = "CROSSENCODER"
#reranker = "SVM"
reranker = "BIENCODER"

# here corpus is the whole corpus, and qrel and query_ds is evaluation datasets, top_100 is initial retrival 
if dataset == "AMAZON":
    prestring = "amazon"
if dataset == "THECA":
    prestring = "theca"
if dataset == "MSMARCO":
    prestring = "msmarco"

corpus_ds = load_from_disk(prestring + "_corpus_ds")
qrel_ds = load_from_disk(prestring + "_qrel_ds")
query_ds = load_from_disk( prestring + "_query_ds_test")
# uncomment if we already have embeddings 
corpus_embeddings = np.load(prestring + "_corpus_embeddings.npy")
query_embeddings = np.load(prestring + "_query_embeddings_test.npy")


# RETRIEVAL
# corpus embeddings are already created
# we just embed the test queries

if RETRIEVAL: 
    from sentence_transformers import SentenceTransformer
    bi_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v2.0", trust_remote_code=True, tokenizer_kwargs={"model_max_length" : 512})
    bi_model.to("cuda")
    # om vi inte har embeddings sen innan kan vi göra de här 
    #corpus_embeddings = bi_model.encode(corpus_ds["combined_text"], show_progress_bar=True) 
    #corpus_embeddings = np.array(corpus_embeddings)
    #np.save(prestring+"_corpus_embeddings", corpus_embeddings)

    query_embeddings = bi_model.encode(query_ds["query"], prompt_name="query", show_progress_bar=True)
    query_embeddings = np.array(query_embeddings)
    np.save(prestring+"_query_embeddings_val", query_embeddings)

    # Create FAISS index (L2 normalized embeddings needed for dot product / cosine similarity)

    index = faiss.read_index(prestring+"_index.faiss")
    dimension = query_embeddings.shape[1]
    #index = faiss.IndexFlatIP(dimension)  # Inner product (equivalent to cosine similarity if normalized)
    #faiss.normalize_L2(corpus_embeddings)
    #index.add(corpus_embeddings)  # Add documents to index# Normalize queries (if using cosine similarity)
    faiss.normalize_L2(query_embeddings)# Perform search (retrieve top k similar docs per query)

    #faiss.write_index(index, prestring+"_index.faiss") #save index to file

    k = 100  # Adjust as needed
    scores, I = index.search(query_embeddings, k)  # D = scores, I = indices of top-k docs
    print("search done")

    # ADD scores to dataset
    query_ids = query_ds["query_id"]  # List of all query IDs
    product_ids = corpus_ds["product_id"]  # List of all product IDs
    query_ids = np.array(query_ids)
    product_ids = np.array(product_ids)

    scores_flat = scores.flatten()  # Convert the tensor to a flattened NumPy array
    query_ids_flat = np.repeat(query_ids, k)
    product_ids_flat = []
    for ind in I.flatten():
        product_id = product_ids[ind]
        product_ids_flat.append(product_id)
    flattened_scores = pd.DataFrame({
        "query_id": query_ids_flat,
        "product_id": product_ids_flat,
        "score": scores_flat
    })

    scores_ds = Dataset.from_pandas(flattened_scores)

    # Tror inte att denna behövs eftersom vi har bara top 100 för varje query eftersom vi hämtar 100 med faiss
    def get_top_k_per_query(scores_ds: Dataset, k: int = 100) -> Dataset:
        scores_df = scores_ds.to_pandas()

        #Sort by 'query_id' and 'score' (descending order)
        scores_df = scores_df.sort_values(by=['query_id', 'score'], ascending=[True, False])

        # Group by 'query_id' and retain top k rows for each query
        top_k_scores_df = (
            scores_df.groupby('query_id')
            .head(k)
            .reset_index(drop=True)
        )

        top_k_scores_ds = Dataset.from_pandas(top_k_scores_df)
        return top_k_scores_ds

    top_100 = get_top_k_per_query(scores_ds,k=100)
    print("top 100 done")

    # ADD LABELS 
    def add_labels(top_100_dataset: Dataset, qrel: Dataset):
        new_col = []
        
        # Create a dictionary from qrel for fast lookup (query_id, product_id) => relevance label
        qrel_dict = {}
        for row in qrel:
            qrel_dict[(row['query_id'], row['product_id'])] = 1  # Relevant products are labeled as 1
        
        # Iterate through the top 100 dataset and add labels
        iterat = top_100_dataset.iter(batch_size=1)
        for i in iterat:
            query_id = i["query_id"][0]
            product_id = i["product_id"][0]
            
            # Check if the (query_id, product_id) exists in qrel_dict
            if (query_id, product_id) in qrel_dict:
                new_col.append(1)  # Label 1 if relevant
            else:
                new_col.append(0)  # Label 0 if not relevant

        # Add the new label column to the dataset
        top_100_dataset = top_100_dataset.add_column("label", new_col)
        
        return top_100_dataset

    top_100_ds = add_labels(top_100, qrel_ds)
    print("labels done")
    """
    ## HERE WE ADD EMBEDDINGS TO TOP_100_DS
    from datasets import concatenate_datasets
    query_id_to_embedding = {q_id: query_embeddings[i] for i, q_id in enumerate(query_ds["query_id"])}
    product_id_to_embedding = {p_id: corpus_embeddings[i] for i, p_id in enumerate(corpus_ds["product_id"])}
    chunk_size = 10000
    chunks = []
    for i in range(0, len(top_100_ds), chunk_size):
        sub_ds = top_100_ds.select(range(i, min(i + chunk_size, len(top_100_ds))))
        q_embs = [query_id_to_embedding[row["query_id"]] for row in sub_ds]
        p_embs = [product_id_to_embedding[row["product_id"]] for row in sub_ds]    
        sub_ds = sub_ds.add_column("query_embedding", q_embs)
        sub_ds = sub_ds.add_column("product_embedding", p_embs)    
        chunks.append(sub_ds)
    top_100_ds = concatenate_datasets(chunks) """

    top_100_ds.save_to_disk(prestring + "_top_ds_val")

else:
    top_100_ds = load_from_disk(prestring + "_top_ds_test")
    top_dict = top_100_ds[:10000]
    top_100_ds = Dataset.from_dict(top_dict)

# NDCG 
# kan ändra här om den ska använda kolonn scores eller ngt annat 
def mrr_score(y_true, y_pred, k):
    all_rr = []
    for i in range(len(y_true)):
        pred = np.array(y_pred[i], dtype='float32')
        true = np.array(y_true[i], dtype='float32')
        data_dict = {"pred" : pred, "true" : true}
        ds = Dataset.from_dict(data_dict)
        ds_sorted = ds.sort("pred", reverse=True)

        pred = ds_sorted["pred"]
        true = ds_sorted["true"]

        true_k = true[:k]
        try:
            rr = 1 / (true_k.index(1)+1)
        except ValueError:
            rr = 0
        all_rr.append(rr)

    mean_rr = sum(all_rr)/(i+1)
    return mean_rr

def map_score(y_true, y_pred, k):
    all_ap = []
    for i in range(len(y_true)):
        pred = np.array(y_pred[i], dtype='float32')
        true = np.array(y_true[i], dtype='float32')
        data_dict = {"pred" : pred, "true" : true}
        ds = Dataset.from_dict(data_dict)
        ds_sorted = ds.sort("pred", reverse=True)

        pred = ds_sorted["pred"]
        true = ds_sorted["true"]
        true_k = true[:k]
        if sum(true_k) == 0:
            app = 0
        else:
            ap = []
            for j in range(len(true_k)):
                p = (sum(true_k[:j+1])*true_k[j])/(j+1)

                ap.append(p)
        
            app = sum(ap)/sum(true_k)
        all_ap.append(app)

    mean_ap = sum(all_ap)/(i+1)
    return mean_ap

def scores_at_k(top_100_labels : Dataset, k, top_k):
    y_true = []
    y_score = []
    score_row = []
    true_row = []
    for i in range(top_100_labels.num_rows):
        score_row.append(top_100_labels[i]["score"])
        true_row.append(top_100_labels[i]["label"])
        if (i+1) % top_k == 0:
            y_true.append(true_row)
            y_score.append(score_row)
            score_row = []
            true_row = []

    y_score = np.array(y_score)
    y_true = np.array(y_true)
    return [ndcg_score(y_true, y_score, k=k), mrr_score(y_true, y_score, k=k), map_score(y_true, y_score, k=k)]
"""
scoress = scores_at_k(top_100_ds, k=10,top_k=100)
print(f"NDCG@10: {scoress[0]}")
print(f"MRR@10: {scoress[1]}")
print(f"MAP@10: {scoress[2]}") """

if RERANK:
    if reranker == "CROSSENCODER":
        from transformers import AutoTokenizer, AutoModel 
        import torch.nn as nn

        #vet inte om vi behöver göra allt detta om vi laddar vår tränade modell, men jag tror det, iallafall klassen
        class SnowflakeClassifier(nn.Module):
            def __init__(self, model_name, num_labels, dropout_value):
                super().__init__() #ksk skicka in model_name
                self.transformer = AutoModel.from_pretrained(model_name, add_pooling_layer=False, trust_remote_code=True)
                self.dropout = nn.Dropout(dropout_value)
                self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)
            
            def forward(self, input_ids, attention_mask, labels=None):
                outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:,0,:]
                cls_embedding = self.dropout(cls_embedding)
                logits = self.classifier(cls_embedding)
                if labels is not None:
                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(logits, labels)
                    return {"loss": loss, "logits": logits}
                return {"logits": logits}

        model_name = "Snowflake/snowflake-arctic-embed-m-v2.0"
        num_labels = 2
        dropout_value = 0.175
        model_cross = SnowflakeClassifier(model_name, num_labels, dropout_value)
        model_cross.load_state_dict(torch.load("training/cross_encoder_temp_8_1.pt", map_location="cuda"))
        model_cross.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        #add combined text to top_100 dataset
        text_dict = {}
        query_dict = {}
        for item in corpus_ds:
            text_dict[item["product_id"]] = item["combined_text"]
        for item in query_ds:
            query_dict[item["query_id"]] = item["query"]

        texts = [text_dict[example["product_id"]] for example in top_100_ds]
        queries = [query_dict[example["query_id"]] for example in top_100_ds]

        top_100_ds = top_100_ds.add_column("combined_text", texts)
        top_100_ds = top_100_ds.add_column("query", queries)

        def crossencoder_reranker(model: nn.Module, tokenizer, top_k: Dataset, device): 
            t_start = time.time()
            query_product_text = []
            query_text = top_k["query"][0] #top_k innehåller bara för en query
            product_texts = top_k["combined_text"]
            for p_text in product_texts:
                query_product_text.append([query_text, p_text])

            tokens = tokenizer(query_product_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
            tokens.to(device)
            model.to(device)

            model.eval()
            with torch.no_grad():
                logits = model(**tokens)["logits"]
                soft = nn.Softmax(dim=-1)
                scores = soft(logits)[:,1]
            t_stop = time.time()
            timed = 1000*(t_stop-t_start)
            return scores,timed
        
        # in rerank we do pairs for each query and its 100 products, one query at a time
        scores = []
        times = []
        k=100
        for i in range(0, len(top_100_ds), k):
            one_top_k = top_100_ds.select(range(i, min(i + 100, len(top_100_ds))))
            one_scores, one_time = crossencoder_reranker(model_cross, tokenizer, one_top_k, device)
            # lägg till scores i list för att sedan lägga till på datasetet
            scores.extend(one_scores.tolist())
            times.append(one_time)
        
        mean_time = sum(times)/len(times)
        no_top_100_ds = top_100_ds.remove_columns("score")
        reranked_top_100_ds = no_top_100_ds.add_column("score", scores)

        reranked_top_100_ds.save_to_disk(prestring+"_"+reranker+"_top_100_reranked_test")
        np.save(prestring+"_"+reranker+"_times.npy", times)    

        scoress = scores_at_k(reranked_top_100_ds,k=10,top_k=100)
        print(f"NDCG@10: {scoress[0]}")
        print(f"MRR@10: {scoress[1]}")
        print(f"MAP@10: {scoress[2]}")
        print(f"Avg time: {mean_time} ms")


    if reranker == "SVM":
        from sklearn import svm

        # load faiss index that we created during retrieval
        index = faiss.read_index(prestring+"_index.faiss")
        #top_100_ds = Dataset.from_dict(top_100_ds[:1000])
  
        query_id_to_embedding = {q_id: query_embeddings[i] for i, q_id in enumerate(query_ds["query_id"])}
        product_id_to_embedding = {p_id: corpus_embeddings[i] for i, p_id in enumerate(corpus_ds["product_id"])}

        def svm_reranker(top_k: Dataset, faiss_index, corpus_embeddings): #top_k får innehålla embeddings
            t_start = time.time()
            # TODO: hämta embeddings från filerna istället för att ha de i top_100_ds
            # kan ändra så att embeddings redan är arrays
            query_embedding = np.array(query_id_to_embedding[top_k["query_id"][0]], dtype="float32") #top_k innehåller bara för en query
            query_embedding = query_embedding[None,...]
            product_embeddings = []
            for item in top_k:
                product_embeddings.append(np.array(product_id_to_embedding[item["product_id"]]))

            #dimension = query_embedding.shape[0]
            #index = faiss.IndexFlatIP(dimension)
            #faiss.normalize_L2(corpus_embeddings)
            #index.add(corpus_embeddings)
            faiss.normalize_L2(query_embedding)
            _, I = faiss_index.search(-1*query_embedding, k)

            negative_embeddings = corpus_embeddings[I] # tror det här ska funka med indexen
            negative_embeddings = negative_embeddings.reshape(k,768)

            x_train = np.concatenate([query_embedding, negative_embeddings])
            x_rank = np.concatenate([query_embedding, product_embeddings])
            y = np.zeros(len(negative_embeddings)+1)
            y[0] = 1

            clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=1, dual='auto', loss="hinge")
            clf.fit(x_train,y)

            scores = clf.decision_function(x_rank)[1:]
            t_stop = time.time()
            timed = 1000*(t_stop-t_start)
            return scores, timed
        
        scores = []
        times = []
        k = 100
        for i in range(0, len(top_100_ds), k):
            one_top_k = top_100_ds.select(range(i, min(i + 100, len(top_100_ds))))
            one_scores, one_time = svm_reranker(one_top_k, index, corpus_embeddings)
            # lägg till scores i list för att sedan lägga till på datasetet
            scores.extend(one_scores)
            times.append(one_time)

        mean_time = sum(times)/len(times)
        no_top_100_ds = top_100_ds.remove_columns("score")
        reranked_top_100_ds = no_top_100_ds.add_column("score", scores)

        reranked_top_100_ds.save_to_disk(prestring+"_"+reranker+"_top_100_reranked_test")
        np.save(prestring+"_"+reranker+"_times.npy", times)    

        scoress = scores_at_k(reranked_top_100_ds,k=10,top_k=100)
        print(f"NDCG@10: {scoress[0]}")
        print(f"MRR@10: {scoress[1]}")
        print(f"MAP@10: {scoress[2]}")
        print(f"Avg time: {mean_time} ms")

    if reranker == "BIENCODER":
        from sentence_transformers import SentenceTransformer
        bi_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0", tokenizer_kwargs={"model_max_length" : 512})
        bi_model.to(device)

        # add combined text to top_100 dataset
        text_dict = {}
        query_dict = {}
        for item in corpus_ds:
            text_dict[item["product_id"]] = item["combined_text"]
        for item in query_ds:
            query_dict[item["query_id"]] = item["query"]

        texts = [text_dict[example["product_id"]] for example in top_100_ds]
        queries = [query_dict[example["query_id"]] for example in top_100_ds]

        top_100_ds = top_100_ds.add_column("combined_text", texts)
        top_100_ds = top_100_ds.add_column("query", queries)

        def biencoder_reranker(model, one_top_k, device):
            t_start = time.time()
            query_embedding = model.encode(one_top_k[0]["query"], prompt_name="query", show_progress_bar=True)
            corpus_embeddings = model.encode(one_top_k["combined_text"], show_progress_bar=True)

            scores = bi_model.similarity(query_embedding, corpus_embeddings)
            t_stop = time.time()
            timed = 1000*(t_stop-t_start)
            return scores, timed

        scores = []
        times = []
        k=100
        for i in range(0, len(top_100_ds), k):
            one_top_k = top_100_ds.select(range(i, min(i + 100, len(top_100_ds))))
            one_scores, one_time = biencoder_reranker(bi_model, one_top_k, device)
            # lägg till scores i list för att sedan lägga till på datasetet
            scores.extend(one_scores.flatten().tolist())
            times.append(one_time)

      
        mean_time = sum(times)/len(times)
        no_top_100_ds = top_100_ds.remove_columns("score")
        reranked_top_100_ds = no_top_100_ds.add_column("score", scores)

        #reranked_top_100_ds.save_to_disk(prestring+"_"+reranker+"_top_100_reranked_test")
        #np.save(prestring+"_"+reranker+"_times.npy", times)    

        scoress = scores_at_k(reranked_top_100_ds,k=10,top_k=100)
        print(f"NDCG@10: {scoress[0]}")
        print(f"MRR@10: {scoress[1]}")
        print(f"MAP@10: {scoress[2]}")
        print(f"Avg time: {mean_time} ms")
