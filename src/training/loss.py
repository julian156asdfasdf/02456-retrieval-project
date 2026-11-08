import torch
import torch.nn as nn


Q = torch.tensor([[1.0,0.0], [0.0,1.0]])
D = torch.tensor([[0.0,2.0], [2.0,0.0]])
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

        #Calculate target vector, which tells the loss function
        #which column contains the correct passage for each query
        #Implementation assumes that query 0 matches passage 0, query 1 matches passage 1, etc.
        batch_size = Q.shape[0]
        target_vec = torch.arange(batch_size)

        #calculate cross entropy
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(Similarity_matrix, target_vec)

        return loss

loss_instance = DPRLoss()
print(loss_instance(Q,D))

