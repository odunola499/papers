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
    "The depletion of the ozone layer poses a serious threat to our planet's atmosphere, requiring urgent international cooperation.",
    "Political unrest in certain regions has led to widespread humanitarian crises, necessitating diplomatic interventions for stability.",
    "Renewable energy sources offer a sustainable solution to reducing carbon emissions, mitigating the impacts of climate change.",
    "Cultural diversity enriches our society, fostering tolerance and understanding among different ethnic groups.",
    "Advancements in medical technology have revolutionized healthcare, improving treatment outcomes and patient care."
]


query = 'Climate change is a pressing global concern that demands immediate action from world leaders'


query_tensors = tokenizer(query, padding = True, truncation = True, max_length = 50, return_tensors='pt')
sentence_tensors = tokenizer(sentences, padding = True, truncation = True, max_length = 50, return_tensors='pt')

with torch.no_grad():
  query_xq = model(**query_tensors)[0][:,0]
  query_xq = pooler(query_xq)
  sentence_xq = model(**sentence_tensors)[0][:,0]
  sentence_xq = pooler(sentence_xq)

sentence_xq = torch.nn.functional.normalize(sentence_xq, p=2, dim=1)
query_xq = torch.nn.functional.normalize(query_xq, p=2, dim=1)

print(cos_sim(query_xq, sentence_xq))