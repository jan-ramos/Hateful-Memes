
import torch 
from transformers import Swinv2Model,AutoImageProcessor
import os 
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from transformers import Swinv2Model,AutoImageProcessor






class Datasetuilder(Dataset):
    def __init__(self, df, tokenizer, sequence_length, 
                 print_text=False):         

        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.print_text = print_text

        model = "microsoft/swinv2-base-patch4-window12-192-22k"
        self.feature_extractor = AutoImageProcessor.from_pretrained(model)
        self.feature_model = Swinv2Model.from_pretrained(model).to('cuda')

        texts = df["text"].values.tolist()
        labels = df["label"].values.tolist()
        images = df["img"].values.tolist()
        ids =  df["idx"].values.tolist()

        self.dataset = []
        for i, inp in enumerate(texts):
            self.dataset.append({"text": inp, "label": labels[i], 'idx': ids[i], 'image': images[i]})
  
    def __len__(self):
        return len(self.dataset)


    def tokenize_data(self, example):
   
        idx = example['idx']
        idx = [idx] if isinstance(idx, str) else idx
        encoded_dict = self.tokenizer(example['text'], padding='max_length', max_length=self.sequence_length, truncation=True, return_tensors='pt')
        tokens = encoded_dict['input_ids']
        token_type_ids = encoded_dict['token_type_ids']
        attn_mask = encoded_dict['attention_mask']
        
        targets = torch.tensor(example['label']).type(torch.int64)

        ## Get Visual Embeddings
        try:
            img = example['image']
            img = Image.open(os.path.join('hateful_memes', img))
            img = np.array(img)
            img = img[...,:3]
            inputs = self.feature_extractor(images=img, return_tensors="pt")
            outputs = self.feature_model(**inputs.to('cuda'))
            visual_embeds = outputs.last_hidden_state
            visual_embeds = visual_embeds.cpu()
        except:
            visual_embeds = np.zeros(shape=(36, 1024), dtype=float)
        

        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.int64)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.int64)

        inputs={"input_ids": tokens.squeeze(),
            "attention_mask": attn_mask.squeeze(),
            "token_type_ids": token_type_ids.squeeze(),
            "visual_embeds": visual_embeds.squeeze(),
            "visual_token_type_ids": visual_token_type_ids.squeeze(),
            "visual_attention_mask": visual_attention_mask.squeeze(),
            "label": targets.squeeze()
        }
        
        return inputs
  
    def __getitem__(self, index):
        inputs = self.tokenize_data(self.dataset[index])
        
        if self.print_text:
            for k in inputs.keys():
                print(k, inputs[k].shape, inputs[k].dtype)

        return inputs