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
import torch.nn.functional as F
import wandb


data = load_data()
huggingface_api = os.environ['HUGGINGFACE_API_KEY']
wandb_api = os.environ['WANDB_API_KEY']
login(token = huggingface_api)
wandb.login(key = wandb_api)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    self.pool.load_state_dict(torch.load('multilingual_embeddings/final_layer.pth', map_location=device))
  def forward(self, input_ids, attention_mask, token_type_ids):
    out = self.base(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)[0][:, 0]
    out = self.pool(out)
    return out

reference = CompositeModel()

student_model = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')

#utils.load_data returns a loader that dishes out source, target pairs. all we have to do is give it a batch size


alpha = 0.5
batch_size = 64
epochs = 1

reference.to(device)
student_model.to(device)
loader = DataLoader(data, batch_size=batch_size)
mse_loss_func = MSELoss()
kld_loss_func = nn.KLDivLoss(reduction = 'batchmean')
optim = torch.optim.Adam(student_model.parameters(), lr=2e-5)

wandb.init(project = "multilingual distillation", entity = 'jenrola2292', name = 'second run')
wandb.config = {
   'learning_rate':2e-5, "epochs": epochs, "batch_size":batch_size, "alpha": alpha
}

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

        mse_mono_loss = mse_loss_func(student_mono_logits, reference_logits)
        mse_english_loss = mse_loss_func(student_mono_logits, student_english_logits)
        mse_loss = (alpha * mse_mono_loss) + ((1 - alpha) * mse_english_loss)

        kld_mono_loss = kld_loss_func(
           F.log_softmax(student_mono_logits, dim = -1), F.softmax(reference_logits, dim = -1)
        )
        kld_english_loss = kld_loss_func(
            F.log_softmax(student_mono_logits, dim = -1), F.softmax(student_english_logits, dim = -1))
        kld_loss = (alpha * kld_mono_loss) + ((1 - alpha) * kld_english_loss)

        loss = (alpha * kld_loss) + ((1 - alpha) * mse_loss)
        train_loss += loss
        loss.backward()
        optim.step()

        wandb.log({'epoch': epoch+1,'batch_loss': loss})
    wandb.log({'epoch': epoch+1, 'epoch_loss': train_loss / len(loader)})
    torch.save(student_model.state_dict(), './model.pt')
    print(f'after {epoch + 1} epochs, loss  is {train_loss / len(loader)}')
    

student_model.push_to_hub('odunola/yoruba-embedding-model-kld')
tokenizer.push_to_hub('odunola/yoruba-embedding-model-kld')




      
      
         
   


