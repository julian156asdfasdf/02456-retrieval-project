import torch
import torch.nn as nn
from src.retrievers.models.encoder import TransformerEncoder


class DPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    
    def forward(self, queries: torch.tensor, documents: torch.tensor):
        """
        @param queries: batch_size x embedding_dim
        @param documents: batch_size x embedding_dim

        Computes cross entropy between the query 

        Returns: A 1d torch tensor 
        """
        #initialize matrices, transpose the documents, 
        #perform matrix multiplication to get similarity matrix
        Q = queries
        D = documents
        D_transposed = D.T
        Similarity_matrix = Q @ D_transposed
        print(Similarity_matrix)

        #Calculate target vector, which tells the loss function
        #which column contains the correct passage for each query
        #Implementation assumes that query 0 matches passage 0, query 1 matches passage 1, etc.
        batch_size = Q.shape[0]
        target_vec = torch.arange(batch_size)

        #calculate cross entropy
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(Similarity_matrix, target_vec)

        return loss

if __name__ == "__main__":
    Q1 = "What is the capital of Denmark?"
    Q2 = "What is the capital of Sweden?"
    Questions = []
    Questions.append(Q1)
    Questions.append(Q2)
    D1 = "Copenhagen is the capital of Denmark"
    D2 = "Stockholm is the capital of Sweden"
    Documents = []
    Documents.append(D1)
    Documents.append(D2)

    #create an instance model of the tranformer, make embeddings of the queries and documents. 
    #print the dimensions. We then calculate the loss, using these embeddings. 
    model = TransformerEncoder(n_layers = 6, n_heads = 4, emb_dim = 768,hidden_dim = 768,feed_forward_dim =  1024, pooling_type = 'max')
        
    Q_embeddings = model.forward(Questions)
    D_embeddings = model.forward(Documents)
    print(Q_embeddings.shape)
    print(D_embeddings.shape)

    loss_instance = DPRLoss()
    print(loss_instance(Q_embeddings,D_embeddings))

