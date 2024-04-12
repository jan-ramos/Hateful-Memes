
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, VisualBertModel

class VisualBERT(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(VisualBERT, self).__init__()
        #configuration = VisualBertConfig.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        #self.visualbert = VisualBertModel.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre', config=configuration)
        
        self.visualbert = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')
        self.embed_cls = nn.Linear(1024,2048)
        self.num_labels = 2
        self.dropout = nn.Dropout(0.3)
        self.cls=  nn.Linear(768, self.num_labels)
        class_weights = [0.77510622, 1.40873991]
        # self.visualbert = VisualBertModel.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre')
        
        self.weight = torch.FloatTensor([class_weights]) #torch.FloatTensor([0.77510622, 1.40873991]),
        nSamples = [5178, 2849]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        self.loss_fct = CrossEntropyLoss(weight=torch.FloatTensor(normedWeights))
        
    def forward(self, tensor):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        visual_embeds_cls = self.embed_cls(tensor['visual_embeds'])
        outputs = self.visualbert(
                tensor['input_ids'],
                attention_mask=tensor['attention_mask'],
                token_type_ids=tensor['token_type_ids'],
                visual_embeds=visual_embeds_cls,
                visual_attention_mask=tensor['visual_attention_mask'],
                visual_token_type_ids=tensor['visual_token_type_ids']
            )
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        reshaped_logits = logits.view(-1, self.num_labels)

        #print('Tensor Value: ',tensor['label'])
        #print('Tensor Shape: ',tensor['label'].shape)
        #print('Tensor View: ',tensor['label'].view(-1))

        loss = self.loss_fct(reshaped_logits, tensor['label'].view(-1))
      
        return loss, reshaped_logits 