
import torch.nn as nn
import torch
from torch.autograd import grad
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
        print(tensor['visual_embeds'].shape)
        visual_embeds_cls = self.embed_cls(tensor['visual_embeds'])
        outputs = self.visualbert(
                tensor['input_ids'],
                attention_mask=tensor['attention_mask'],
                token_type_ids=tensor['token_type_ids'],
                visual_embeds=visual_embeds_cls,
                visual_attention_mask=tensor['visual_attention_mask'],
                visual_token_type_ids=tensor['visual_token_type_ids'],
                output_hidden_states=True,
                return_dict = True
            )
        print(visual_embeds_cls.shape)
        #for x in outputs['hidden_states']:
            #print(x.shape)
        #print(outputs['pooler_output'].shape)
        #outputs = output_dict.
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        #print(logits)
        reshaped_logits = logits.view(-1, self.num_labels)

        #print('Tensor Value: ',tensor['label'])
        #print('Tensor Shape: ',tensor['label'].shape)
        #print('Tensor View: ',tensor['label'].view(-1))

        loss = self.loss_fct(reshaped_logits, tensor['label'].view(-1))
      
        if pooled_output.requires_grad:
            #perturbed_sentence = self.adv_attack(outputs['hidden_states'][0], loss)
            perturbed = self.adv_attack(visual_embeds_cls, loss)
            mask =(tensor['attention_mask'], tensor['visual_attention_mask'])
            #print(mask.shape)
            perturbed = (outputs['hidden_states'][0], perturbed)
            adv_loss = self.adversarial_loss(perturbed, mask, tensor['label'])
            pass


        return loss, reshaped_logits 
    
    def adv_attack(self, emb, loss, epsilon=0.05):
        loss_grad = grad(loss, emb, retain_graph=True)[0]
        loss_grad_norm = torch.sqrt(torch.sum(loss_grad**2, (1,2)))
        perturbed_sentence = emb + epsilon * (loss_grad/(loss_grad_norm.reshape(-1,1,1)))
        return perturbed_sentence

    def adversarial_loss(self, perturbed, attention_mask, labels):
        #print(perturbed.shape)
        #extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        #extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        #ve = self.embed_cls(perturbed[:, 512:548, :])
        out = self.visualbert(inputs_embeds=perturbed[0][:, 0:512, :], attention_mask=attention_mask[0], visual_embeds=perturbed[1], visual_attention_mask=attention_mask[1])
        print(out)
        encoded_layers_last = out['pooler_output']
        encoded_layers_last = self.dropout(encoded_layers_last)
        logits = self.cls(encoded_layers_last)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        adv_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return adv_loss