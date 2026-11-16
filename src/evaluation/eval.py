import torch
import torch.nn as nn
from src.training.loss import DPRLoss
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

class evaluator:
    def __init__(self, encoder, batch_size, save_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.encoder.load_state_dict(torch.load(save_dir + r'\encoder.pt'))
        self.batch_size = batch_size


    def encode(self, texts):
        vecs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size)):
                vecs.append(self.encoder(texts[i:i + self.batch_size]).cpu())
        return torch.cat(vecs)    


    def eval(self, queries, documents, k=10):
        doc_ids   = [id   for ids   in documents["document_id"] for id   in ids]    # flatten: each row is a list
        doc_texts = [text for texts in documents["text"]        for text in texts]

        d_vecs = self.encode(doc_texts)                        
        q_vecs = self.encode([q.text for q in queries])

        scores = q_vecs @ d_vecs.T                            
        top = scores.topk(k, dim=1).indices              

        recall = 0.0
        for q, row in zip(queries, top):
            retrieved = {doc_ids[j] for j in row}
            relevant = set(q.relevant_document_ids)
            recall += len(retrieved & relevant) / len(relevant)

        print(f"recall {k}: {recall / len(queries):.4f}")



        