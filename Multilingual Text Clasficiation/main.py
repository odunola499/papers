from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from utils import load_data, calculate_accuracy, tokenizer
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from huggingface_hub import login
import wandb
import os
from tqdm.auto import tqdm


huggingface_api = os.environ['HUGGINGFACE_API_KEY']
wandb_api = os.environ['WANDB_API_KEY']
login(token = huggingface_api)
wandb.login(key = wandb_api)

batch_size = 16
epochs = 3
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_url = 'bert-base-multilingual-uncased'
model = AutoModelForSequenceClassification.from_pretrained(model_url, num_labels= 3).to(device)


train_data, valid_data, test_data = load_data()

train_loader = DataLoader(train_data, batch_size=batch_size)
valid_loader = DataLoader(valid_data, batch_size = batch_size)
test_loader = DataLoader(test_data, batch_size = batch_size)

wandb.init(project = "Multilingual Text Classification", entity = 'jenrola2292', name = 'first run')
wandb.config = {
   'learning_rate':2e-5, "epochs": epochs, "batch_size":batch_size
}

optimizer = Adam(model.parameters(), lr = 2e-5)
loss_func = CrossEntropyLoss()
model.to(device)


for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)
        logits = model(input_ids, attention_mask)[0]
        loss = loss_func(logits, label)
        train_loss += loss.item()
        accuracy = calculate_accuracy(logits, label)
        train_accuracy += accuracy
        loss.backward()
        optimizer.step()
        wandb.log({'epoch': epoch+1,'training_loss': loss, 'validation_accuracy': accuracy})

    valid_loss = 0.0
    valid_accuracy = 0.0
    model.eval()
    for batch in tqdm(valid_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)
        logits = model(input_ids, attention_mask)[0]
        loss = loss_func(logits, label)
        valid_loss += loss.item()
        accuracy = calculate_accuracy(logits, label)
        valid_accuracy += accuracy
        wandb.log({'epoch': epoch+1,'validation_loss': loss, 'validation_accuracy': accuracy})

    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)

    print(f"After {epoch + 1} : Train Loss {train_loss} : Valid Loss {valid_loss}")


model.push_to_hub('odunola/multilingual_text_classifier')
tokenizer.push_to_hub('odunola/multilingual_text_classifier')