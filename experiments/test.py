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


tokenizer = 'google-bert/bert-base-uncased'
model = "microsoft/swinv2-base-patch4-window12-192-22k"
dataset = DatasetBuilder(model,tokenizer,'train.jsonl')


visual_model = VisualBERT()
visual_model.forward(dataset[0])