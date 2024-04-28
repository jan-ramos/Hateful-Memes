from data_setup.databuilder import DatasetBuilder
import torch 
from transformers import Swinv2Model,AutoImageProcessor
import os 
import pandas as pd
import numpy as np
from model.visbert import VisualBERT
from sklearn.metrics import roc_auc_score
from datasets import list_metrics, load_metric
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from datasets import load_metric
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import TrainingArguments, Trainer
import os



batch_size = 24
seq_len = 50


parent_directory = os.getcwd()
data_directory = os.path.join(parent_directory,'data')
df = pd.read_csv(data_directory + 'data.csv')
df_train = df[:8500]
df_val = df[8500:9540]
df_test = df[9540:]

df_train['text_len'] = df_train['text'].str.split().str.len()
df_train['idx'] = df_train['id'].astype(str).str.zfill(5)
df_val['idx'] = df_val['id'].astype(str).str.zfill(5)
df_test['idx'] = df_test['id'].astype(str).str.zfill(5)


metrics_list = list_metrics()
print(metrics_list)

acc_metric = load_metric('accuracy')
f1_metric = load_metric('f1')
precision_metric = load_metric('precision')
recall_metric = load_metric('recall')


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

model = "microsoft/swinv2-base-patch4-window12-192-22k"
feature_extractor = AutoImageProcessor.from_pretrained(model)
feature_model = Swinv2Model.from_pretrained(model).to('cuda')

dataset = DatasetBuilder(df_val, tokenizer, 50, True)

model = VisualBERT().to('cuda')



from transformers import TrainingArguments, Trainer
batch_size = 24
seq_len = 50

metric_name = "auroc"

args = TrainingArguments(
    output_dir = "test",
    seed = 4, 
    evaluation_strategy = "steps",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs= 5,
    weight_decay=0.05,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    eval_steps = 50,
    save_steps = 500,
    fp16 = False,
    gradient_accumulation_steps = 2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)
    auc_score = roc_auc_score(labels, predictions)
    
    return {"accuracy": acc['accuracy'], "auroc": auc_score,'f1':f1['f1'],'precision':precision['precision'],'recall':recall['recall']} 


trainer = Trainer(
    model,
    args,
    train_dataset = DatasetBuilder(df_train,tokenizer=tokenizer, sequence_length=seq_len),
    eval_dataset =  DatasetBuilder(df_val,tokenizer=tokenizer, sequence_length=seq_len),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()