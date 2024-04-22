
import torch 
from transformers import BertTokenizer, VisualBertModel
from transformers import Swinv2Model,AutoImageProcessor
import os 
import pandas as pd
import jsonlines
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available else 'cpu'
print(device)

class DatasetBuilder():
    def __init__(self,model = None,tokenizer = None,json_file = None):
        #print(tokenizer)
        self.parent_directory = os.getcwd()
        self.data_directory = os.path.join(self.parent_directory,'data')
        self.json_file = json_file
        #print(os.getcwd())
        self.tokenizer = BertTokenizer.from_pretrained("./tokenizer/", local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained(model)
        self.viz_model = Swinv2Model.from_pretrained(model).to(device)
        self.dataset = self.create_dataframe()

    def create_dataframe(self):
        dataset = pd.DataFrame(columns=['id','img','label','text'])
        with jsonlines.open(os.path.join(self.data_directory,self.json_file)) as reader:
            for data in reader:
                data_df = pd.DataFrame([data])
                dataset = pd.concat([dataset,data_df]).reset_index(drop=True)

        dataset['img'] = (self.data_directory+'/'+dataset['img']).apply(self.load_image)
        return dataset
    

    def tokenize_data(self,value):
        inputs = self.tokenizer(value['text'], return_tensors='pt', padding='max_length', max_length=36, truncation=True)
        target = torch.tensor(value['label']).type(torch.int64)
        
        
        image = self.image_processor(value['img'], return_tensors="pt")
            
        try:
            outputs = self.viz_model(**image)
            visual_embeds = outputs.last_hidden_state
        except:
            visual_embeds = np.zeros(shape=(36, 1024), dtype=float)

            
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.double)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.double)
        inputs.update(
            {
                "visual_embeds": visual_embeds.squeeze(),
                "visual_token_type_ids": visual_token_type_ids.squeeze(),
                "visual_attention_mask": visual_attention_mask.squeeze(),
                "label":target.to(device)
            }
        )

        return inputs
    
    def get_dataset(self):
        return self.dataset
  
    def __getitem__(self, index):
        inputs = self.tokenize_data(self.dataset.loc[index])
        
        #for k in inputs.keys():
            #print(k, inputs[k].shape, inputs[k].dtype)

        return inputs

    def __len__(self):
        return len(self.dataset)

    def load_image(self,filepath):
        image = Image.open(filepath).convert('RGB')
        image_array = np.array(image)
        return image_array