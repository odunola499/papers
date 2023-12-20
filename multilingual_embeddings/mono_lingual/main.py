from torch import nn
import torch
from transformers import AutoTokenizer, AutoModel
#let us say that we have sentence pairs of english and yoruba 
#first we load the main model
import os
from huggingface_hub import login
from torch.utils.data import DataLoader
from utils import load_data
from tqdm.auto import tqdm
from torch.nn import MSELoss
data = load_data()
huggingface_api = os.environ['HUGGINGFACE_API_KEY']

login(token = huggingface_api)

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
class Pooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1024, 768)
        self.a = nn.Linear(768, 768, bias = True)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.layer(x)
        x = self.a(x)
        x = self.activation(x)
        return x
    
class CompositeModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.base = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
    self.pool = Pooler()
    self.pool.load_state_dict(torch.load('multilingual_embeddings/final_layer.pth'))
  def forward(self, input_ids, attention_mask, token_type_ids):
    out = self.base(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)[0][:, 0]
    out = self.pool(out)
    return out

reference = CompositeModel()

student_model = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')

#utils.load_data returns a loader that dishes out source, target pairs. all we have to do is give it a batch size


alpha = 0.5
batch_size = 64
epochs = 3
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
reference.to(device)
student_model.to(device)
loader = DataLoader(data, batch_size=batch_size)
loss_func = MSELoss()
optim = torch.optim.Adam(student_model.parameters(), lr=2e-5)



for epoch in range(epochs):
    train_loss =0.0
    for batch in tqdm(loader, leave = True):
        optim.zero_grad()
        english_ids = batch['english_ids'].to(device)
        english_mask = batch['english_mask'].to(device)
        english_token_ids = batch['english_token_ids'].to(device)
        mono_ids = batch['mono_ids'].to(device)
        mono_mask = batch['mono_mask'].to(device)
        mono_token_ids = batch['mono_token_ids'].to(device)
        with torch.no_grad():
            reference_logits = reference(english_ids, english_mask, english_token_ids)
        student_mono_logits = student_model(mono_ids, mono_mask, mono_token_ids)[0][:,0]
        student_english_logits = student_model(english_ids, english_mask, english_token_ids)[0][:,0]
        mono_loss = loss_func(student_mono_logits, reference_logits)
        english_loss = loss_func(student_mono_logits, student_english_logits)
        loss = (alpha * mono_loss) + ((1 - alpha) * english_loss)
        train_loss += loss
        loss.backward()
        optim.step()
    print(f'after {epoch + 1} epochs, loss  is {loss}')
    

student_model.push_to_hub('odunola/yoruba-embedding-model')
tokenizer.push_to_hub('odunola/yoruba-embedding-model')




      
      
         
   


