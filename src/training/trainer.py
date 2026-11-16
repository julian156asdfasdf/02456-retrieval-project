import torch
import torch.nn as nn
from src.training.loss import DPRLoss
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, encoder, lr=1e-4, batch_size=32, epochs=1, save_path = r'C:\Users\julia\Desktop\3_semester\deep learning\02456-retrieval-project\outputs\encoder\checkpoints'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.batch_size = batch_size
        self.epochs = epochs
        self.criterion = DPRLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
        self.save_path = save_path


    def train(self, train_queries, val_queries, documents, k):
        self.doc_text = {}
        for ids, texts in zip(documents["document_id"], documents["text"]):
            for id, text in zip(ids, texts):
                self.doc_text[id] = text


        for epoch in range(self.epochs):
            self.encoder.train()
            train_loss = self.one_train_epoch(train_queries)
            self.encoder.eval()
            val_recall = self.one_validation_epoch(val_queries, documents, k)


            print(f"epoch {epoch + 1} loss {train_loss} validation top-{k} recall: {val_recall}")

        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), self.save_path + r'\encoder.pt')


    def one_train_epoch(self, queries):

        random.shuffle(queries)
        total, n_batches = 0.0, 0
        for i in tqdm(range(0, len(queries), self.batch_size)):
            batch = queries[i:i+self.batch_size]

            q_texts = [q.text for q in batch]
            ids = [random.choice([d for d in q.relevant_document_ids]) for q in batch]
            d_texts = [self.doc_text[d] for d in ids]

            q_vecs = self.encoder(q_texts).to(self.device)
            d_vecs = self.encoder(d_texts).to(self.device)

            loss = self.criterion(q_vecs, d_vecs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total += loss.item()
            n_batches += 1
        return loss


    def one_validation_epoch(self, val_queries, documents, k):
        doc_ids = [id for ids in documents["document_id"] for id in ids]  
        doc_texts = [text for texts in documents["text"] for text in texts]

        d_vecs = self.encode(doc_texts)                        
        q_vecs = self.encode([q.text for q in val_queries])

        scores = q_vecs @ d_vecs.T                            
        top = scores.topk(k, dim=1).indices              

        recall = 0.0
        for q, row in zip(val_queries, top):
            retrieved = {doc_ids[j] for j in row}
            relevant = set(q.relevant_document_ids)
            recall += len(retrieved & relevant) / len(relevant)

        return recall/ len(val_queries)



    def encode(self, texts):
        vecs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size)):
                vecs.append(self.encoder(texts[i:i + self.batch_size]).cpu())
        return torch.cat(vecs)    



