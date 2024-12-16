import torch
import torch.nn as nn
import os
import timm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.backends.cudnn as cudnn
from utils.libs import *


class Traffic3DVG(nn.Module):

    def __init__(self, parser):
        super(Traffic3DVG, self).__init__()
        self.parser = parser
        self.grad_clip = parser.grad_clip
        self.text_tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese")
        self.text_encoder = AutoModelForSequenceClassification.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese")
        self.state_encoder = nn.Sequential(nn.Linear(9, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, self.parser.d_model))
        self.img_encoder = timm.create_model('resnet50', pretrained=True)
        self.down = nn.Linear(self.parser.d_model * 4, self.parser.d_model)
        self.img_state_fusion = Img_State_Fusion(self.parser.d_model, 1024, self.parser.d_model)
        self.mm_fusion = nn.Sequential(nn.Linear(self.parser.d_model * 2, self.parser.d_model), nn.ReLU(), nn.Dropout(0.3), nn.Linear(self.parser.d_model, self.parser.d_model))
        self.classifier = nn.Sequential(nn.Linear(self.parser.d_model, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
         
        cudnn.benchmark = True
        
        # freeze parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False      
            
    def forward(self, images, states, queries):
        image_features = self.img_encoder.forward_features(images)
        image_features = self.img_encoder.global_pool(image_features)
        image_features = self.down(image_features)
        states_embeddings = self.state_encoder(states)      # (2*bs, d_model)
        visual_features = self.img_state_fusion(image_features, states_embeddings)  # (2*bs, d_model)
        queries_inputs = self.text_tokenizer(queries, return_tensors='pt', padding=True)  
        queries_inputs = {k: v.to(self.parser.device) for k, v in queries_inputs.items()} 
        queries_features = self.text_encoder(**queries_inputs).logits
        # alignment
        image_normalized = F.normalize(visual_features, dim=1)              # (2*bs, d_model)
        text_normalized = F.normalize(queries_features, dim=1)                # (2*bs, d_model)
        # fusion
        joint_embeddings= self.mm_fusion(torch.cat((image_normalized, text_normalized), dim=1))  # (2*bs, d_model)
        logits = self.classifier(joint_embeddings).squeeze(dim=1)
        return image_normalized, text_normalized, logits