from torch import nn
import torch
from transformers import AutoModel
#let us say that we have sentence pairs of english and yoruba 
#first we load the main model
import os
from huggingface_hub import login
from torch.utils.data import DataLoader
from utils import load_data, tokenizer
from tqdm.auto import tqdm
import torch.nn.functional as F
import wandb


data = load_data()
huggingface_api = os.environ['HUGGINGFACE_API_KEY']
wandb_api = os.environ['WANDB_API_KEY']
login(token = huggingface_api)
wandb.login(key = wandb_api)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = AutoModel.from_pretrained('odunola/bge-base-en-v1.5-yoruba')

batch_size = 16
loader = DataLoader(data, batch_size = batch_size)

cos_sim = nn.CosineSimilarity().to(device)
loss_func = nn.CrossEntropyLoss().to(device)
model.to(device)
scale = 20.0
optim = torch.optim.Adam(model.parameters(), lr=2e-5)
epochs = 1

wandb.init(project = "multilingual distillation", entity = 'jenrola2292', name = 'final run')
wandb.config = {
   'learning_rate':2e-5, "epochs": epochs, "batch_size":batch_size
}


for epoch in range(epochs):
    loop = tqdm(loader, leave = True)
    for batch in loop:
        optim.zero_grad()
        anchor_ids = batch['anchor_ids'].to(device)
        anchor_mask = batch['anchor_mask'].to(device)
        anchor_token_ids = batch['anchor_token_ids'].to(device)
        pos_ids = batch['positive_ids'].to(device)
        pos_mask = batch['positive_mask'].to(device)
        pos_token_ids = batch['positive_token_ids'].to(device)
        
        a = model(
            anchor_ids, anchor_mask, anchor_token_ids
        )[0][:,0]  # all token embeddings
        p = model(
            pos_ids, pos_mask, pos_token_ids
        )[0][:,0]

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

        wandb.log({'epoch': epoch+1,'batch_loss': loss})
    torch.save(model.state_dict(), './model.pt')


model.push_to_hub('odunola/yoruba-embedding-model-kld')
tokenizer.push_to_hub('odunola/yoruba-embedding-model-kld')

