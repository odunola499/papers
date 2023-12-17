from utils import load_data
import os
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import torch

from tqdm.auto import tqdm
import scipy



dataset = load_data()
from huggingface_hub import login



model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
                                  
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
    

pooler = Pooler()

#now we have our model

#we need to freeze all layers except the pooling module
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
for name, param in model.named_parameters():
    if 'pooler' not in name:
        param.__requires_grad = False  #freezing layes

#now we need to finetune the model

batch_size = 32
loader = DataLoader(dataset, batch_size = batch_size)

cos_sim = nn.CosineSimilarity().to(device)
loss_func = nn.CrossEntropyLoss().to(device)
model.to(device)
scale = 20.0
optim = torch.optim.Adam(model.parameters(), lr=2e-5)
epochs = 2
pooler.to(device)

for epoch in range(epochs):
    loop = tqdm(loader, leave = True)
    for batch in loop:
        optim.zero_grad()
        anchor_ids = batch['anchor_ids'].to(device)
        anchor_mask = batch['anchor_mask'].to(device)
        pos_ids = batch['positive_ids'].to(device)
        pos_mask = batch['positive_mask'].to(device)
        with torch.no_grad(): #we are not changing weights of base model
            a = model(
                anchor_ids, attention_mask=anchor_mask
            )[0][:,0]  # all token embeddings
            p = model(
                pos_ids, attention_mask=pos_mask
            )[0][:,0]
        a = pooler(a)
        p = pooler(p)
        
        scores = torch.stack([
            cos_sim(
                a_i.reshape(1, a_i.shape[0]), p
            ) for a_i in a])

        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        # and now calculate the loss
        loss = loss_func(scores*scale, labels)
        # using loss, calculate gradients and then optimize
        loss.backward()
        optim.step()



torch.save(pooler.state_dict(), 'final_layer.pth')

