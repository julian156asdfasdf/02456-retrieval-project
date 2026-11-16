import torch
import torch.nn as nn
from torch.optim import Adam
from src.training.loss import DPRLoss
from torch.utils.data import TensorDataset, DataLoader
from src.retrievers.models.encoder import TransformerEncoder

batch_size = 2

#prepare the test data, format it, instantiate two models; one for the queries and one for the documents. 
pairs = [("What is the capital of Denmark?", "Copenhagen is the capital of Denmark"), ("What is the capital of Sweden?", "Stockholm is the capital of Sweden")]
DataloaderNew = DataLoader(pairs, batch_size=batch_size, shuffle=False)
model = TransformerEncoder(n_layers = 6, n_heads = 4, emb_dim = 768,hidden_dim = 768,feed_forward_dim =  1024, pooling_type = 'max')
model_2 = TransformerEncoder(n_layers = 6, n_heads = 4, emb_dim = 768,hidden_dim = 768,feed_forward_dim =  1024, pooling_type = 'max')




class Trainer:
    def __init__(self):
        self.criterion = DPRLoss()

    def train(self, query_encoder: nn.Module, document_encoder: nn.Module, dataLoader):
        #Create adam optimizer
        optimizer = Adam(list(query_encoder.parameters()) + list(document_encoder.parameters()), lr=1e-4)
        #create artificial test training with x epochs. 
        for epoch in range(10):
            for query, document in dataLoader:
                query_embeddings = query_encoder(query)
                document_embeddings = document_encoder(document)
                #calculate loss
                loss = self.criterion(query_embeddings,document_embeddings)
                #Set gradients to zero, so we dont retain old gradients and therefore accumulate gradients.
                optimizer.zero_grad()
                #computes the gradients
                loss.backward()
                print(loss)
                #updates the weights 
                optimizer.step()
            


#creates an instance we can train on. we then run the train method using our transformer embeddings on our testdata. 
trainer_instance = Trainer()
trainer_instance.train(query_encoder=model,document_encoder=model_2,dataLoader=DataloaderNew)