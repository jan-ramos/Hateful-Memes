
import torch.nn as nn
import torch
from torch.autograd import grad
from torch.nn import CrossEntropyLoss
from transformers import VisualBertModel
from transformers import VisualBertModel,VisualBertConfig


device ='cuda' if torch.cuda.is_available else 'cpu'



class VisualBERT(torch.nn.Module):
    def __init__(self):
        """
        Steps:
           1. Initialize pre-trained VisualBERT model.
           2. Initialize two nn.Linear layers to get desired tensor outputs.
           3. Initialize weights from weighted mean: [0.77510622, 1.40873991]

        """
        super(VisualBERT, self).__init__()
        configuration = VisualBertConfig.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre',
                                                hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        self.visualbert = VisualBertModel.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre', config=configuration)
        self.embed_cls = nn.Linear(1024, 1024)
        self.num_labels = 2
        self.dropout = nn.Dropout(0.3)
        self.cls=  nn.Linear(768, self.num_labels)
        self.weight = torch.FloatTensor([0.77510622, 1.40873991]) 
        nSamples = [5178, 2849]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        self.loss_fct = CrossEntropyLoss(weight=torch.FloatTensor(normedWeights))


    def forward(self, input_ids, attention_mask, token_type_ids, visual_embeds, visual_attention_mask, visual_token_type_ids, labels):
   
        visual_embeds_cls = self.embed_cls(visual_embeds)

        outputs = self.visualbert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                visual_embeds=visual_embeds_cls,
                visual_attention_mask=visual_attention_mask,
                visual_token_type_ids=visual_token_type_ids,
            )
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        reshaped_logits = logits.view(-1, self.num_labels)

        loss = self.loss_fct(reshaped_logits, labels.view(-1))
      

        if pooled_output.requires_grad:

            perturbed_sentence = self.adv_attack(outputs.last_hidden_state, loss)

            perturbed = self.adv_attack(visual_embeds_cls, loss)
            
            mask =(attention_mask, visual_attention_mask)

            perturbed = (perturbed_sentence, perturbed)
      
            adv_loss = self.adversarial_loss(perturbed, input_ids,visual_embeds_cls, attention_mask, mask, labels, token_type_ids,visual_token_type_ids)
            loss = loss + adv_loss
            
        

        return loss, reshaped_logits


    
    def adv_attack(self, emb, loss, epsilon=0.05):
        loss_grad = grad(loss, emb, retain_graph=True)[0]
        loss_grad_norm = torch.sqrt(torch.sum(loss_grad**2, (1,2)))
        perturbed_sentence = emb + epsilon * (loss_grad/(loss_grad_norm.reshape(-1,1,1)))


        return perturbed_sentence

    def adversarial_loss(self, perturbed, input_ids, visual_embeds_cls, attention_mask, mask, labels, token_type_ids,visual_token_type_ids):
        out = self.visualbert(
                inputs_embeds = perturbed[0][:, :50, :],
                attention_mask=attention_mask
                ,token_type_ids=token_type_ids
                ,visual_embeds = perturbed[1]
                ,visual_attention_mask=mask[1]
                ,visual_token_type_ids=visual_token_type_ids
            )
    
        encoded_layers_last = out['pooler_output']
        encoded_layers_last = self.dropout(encoded_layers_last)
        logits = self.cls(encoded_layers_last)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        adv_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return adv_loss