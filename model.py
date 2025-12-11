import torch
from transformers import AutoTokenizer, AutoModel

class CustomModel(torch.nn.Module):
    def __init__(self, model_name):
        super(CustomModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.linear = torch.nn.Linear(self.base_model.config.hidden_size, 1) 
        

    def forward(self, input_ids, attention_mask, **keywords):
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        x = self.linear(hidden).squeeze(-1)  
        
        return x