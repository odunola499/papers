import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer



model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
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


pooler = Pooler()
pooler.load_state_dict(torch.load('final_layer.pth'))
cos_sim = nn.CosineSimilarity()
#you can change values of sentences an query
sentences = [
    "AI has revolutionized various industries, transforming how we interact with modern technology.",
    "The migration patterns of birds have fascinated scientists for centuries, revealing complex behaviors in nature.",
    "From ancient civilizations to modern society, the evolution of transportation has shaped our way of life.",
    "The culinary arts have always been a source of cultural pride, offering a diverse tapestry of flavors from around the world.",
    "The exploration of space continues to captivate our imaginations, pushing the boundaries of scientific discovery."
]

query = 'The advancement of artificial intelligence has transformed numerous industries, revolutionizing how we interact with technology'


query_tensors = tokenizer(query, padding = True, truncation = True, max_length = 50, return_tensors='pt')
sentence_tensors = tokenizer(sentences, padding = True, truncation = True, max_length = 50, return_tensors='pt')

with torch.no_grad():
  query_xq = model(**query_tensors)[0][:,0]
  query_xq = pooler(query_xq)
  sentence_xq = model(**sentence_tensors)[0][:,0]
  sentence_xq = pooler(sentence_xq)

sentence_xq = torch.nn.functional.normalize(sentence_xq, p=2, dim=1)
query_xq = torch.nn.functional.normalize(query_xq, p=2, dim=1)

cos_sim(query_xq, sentence_xq)