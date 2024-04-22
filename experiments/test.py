from data_setup.databuilder import DatasetBuilder

import torch 
from transformers import BertTokenizer, VisualBertModel
from transformers import Swinv2Model,AutoImageProcessor
import os 
import pandas as pd
import jsonlines
from PIL import Image
import numpy as np
from model.visbert import VisualBERT
from sklearn.metrics import roc_auc_score
from datasets import list_metrics, load_metric
from transformers import TrainingArguments, Trainer
batch_size = 24
seq_len = 50

metric_name = "auroc"

args = TrainingArguments(
    output_dir = "model-checkpoint",
    seed = 110, 
    evaluation_strategy = "steps",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs= 10,
    weight_decay=0.05,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    eval_steps = 1,
    save_steps = 500,
    fp16 = False,
    gradient_accumulation_steps = 2
)




#metrics_list = list_metrics()
#print(metrics_list)


acc_metric = load_metric('accuracy')
f1_metric = load_metric('f1')
precision_metric = load_metric('precision')
recall_metric = load_metric('recall')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)
    auc_score = roc_auc_score(labels, predictions)
    return {"accuracy": acc['accuracy'], "auroc": auc_score,'f1':f1['f1'],'precision':precision['precision'],'recall':recall['recall']}




tokenizer = 'google-bert/bert-base-uncased'
model = "microsoft/swinv2-base-patch4-window12-192-22k"
#dataset = DatasetBuilder(model,tokenizer,'train.jsonl')



visual_model = VisualBERT()
swin_model = "microsoft/swinv2-base-patch4-window12-192-22k"
#visual_model.forward(dataset[0])

optimizer = torch.optim.Adam(visual_model.parameters(), )

trainer = Trainer(
    visual_model,
    args,
    train_dataset = DatasetBuilder(swin_model,tokenizer,json_file='train.jsonl'),
    eval_dataset = DatasetBuilder(swin_model,tokenizer,json_file='test.jsonl'),
    #tokenizer=BertTokenizer.from_pretrained("./tokenizer/", local_files_only=True),
    compute_metrics=compute_metrics
)


trainer.train()
#for i in range(1):
##    for j in dataset.get_dataset():
 #       optimizer.zero_grad()
  #      loss, _ = visual_model(j)
 #       loss.backward()
 #       optimizer.step()
 #       print(loss)
