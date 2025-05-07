from transformers import AutoTokenizer, AutoModel
import torch 
import torch.nn as nn
from datasets import Dataset, load_from_disk
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm

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
    return np.array([ndcg_score(y_true, y_score, k=k), mrr_score(y_true, y_score, k=k), map_score(y_true, y_score, k=k)])


model_name = "Snowflake/snowflake-arctic-embed-m-v2.0"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# have to run this to load our pretrained classifier?
num_labels = 2
dropout_value = 0.175
model_cross = SnowflakeClassifier(model_name, num_labels, dropout_value)
device = "cuda"

# load top 100 from retrieval ÄNDRA
top_100_labels_test = load_from_disk("small_TEST")
text = top_100_labels_test["query_product_text"]

model_cross.load_state_dict(torch.load("training/cross_encoder_temp_8_6.pt", map_location="cuda"))
model_cross.to(device)

# predict scores
model_cross.eval()
logits = []
batch_size = 128
with torch.no_grad():
    for i in tqdm(range(0, len(top_100_labels_test), batch_size)):
        if i+batch_size>len(top_100_labels_test)-1:
            batch = text[i:]
        else:
            batch = text[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        outputs = model_cross(**tokens)
        logits.append(outputs["logits"].to("cpu"))
logits = torch.cat(logits)
soft = nn.Softmax(dim=-1)
scores = soft(logits)
rerank_scores = scores[:,1]
np.save("rerank_scores_best", rerank_scores)

"""
# print score
temp_dataset = top_100_labels_test.remove_columns("score")
test_dataset_labels_scores = temp_dataset.add_column("score", rerank_scores.tolist())
print("starting ndcg")
scoress = scores_at_k(test_dataset_labels_scores, k=10,top_k=100)
print(f"NDCG@10: {scoress[0]}")
print(f"MRR@10: {scoress[1]}")
print(f"MAP@10: {scoress[2]}")
"""

