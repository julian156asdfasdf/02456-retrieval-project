import torch
import torch.nn as nn
from torch.optim import Adam
from src.training.loss import DPRLoss
from torch.utils.data import TensorDataset, DataLoader

batch_size = 2
Q = torch.tensor([[1.0,0.0], [0.0,1.0]])
D = torch.tensor([[0.0,2.0], [2.0,0.0]])
dataset = TensorDataset(Q,D)
Dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

query_encoder = nn.Linear(Q.shape[1], 2)
document_encoder = nn.Linear(D.shape[1], 2)



class Trainer:
    def __init__(self):
        self.criterion = DPRLoss()

    def train(self, query_encoder: nn.Module, document_encoder: nn.Module, dataLoader):
        optimizer = Adam(list(query_encoder.parameters()) + list(document_encoder.parameters()), lr=1e-3)
        for query, document in dataLoader:
            query_embeddings = query_encoder(query)
            document_embeddings = document_encoder(document)
            loss = self.criterion(query_embeddings,document_embeddings)
            optimizer.zero_grad()
            loss.backward()
            print(loss)
            optimizer.step()
            


trainer_instance = Trainer()
trainer_instance.train(query_encoder=query_encoder,document_encoder=document_encoder,dataLoader=Dataloader)
