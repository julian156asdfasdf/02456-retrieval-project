import torch
import torch.nn as nn


class DPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def forward(self, queries: torch.tensor, documents: torch.tensor):
        sim_matrix = (queries @ documents.T)

        # We can rewrite the equation from the project description
        # With ln(a/b) = ln(a) - ln(b)
        # Also are we supposed to two seperate encoders? one for query and one for documents?
        positives = torch.diag(sim_matrix)
        
        loss = -positives + torch.logsumexp(sim_matrix, dim=1)
        return torch.mean(loss)
