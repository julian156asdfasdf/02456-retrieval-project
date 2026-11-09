import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int, batch_size=32):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.batch_size = batch_size

        self.setup_model()


    def setup_model(self):
        """
        Instantiate the model.
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
        self.embeddings = nn.Embedding(30522, self.emb_dim)
        self.pe = PositionalEncoding(self.emb_dim, 0.1)
        self.multi_headed_attention = MultiHeadedAttention(self.n_heads, self.emb_dim)
        self.layer_normalizer_1 = nn.LayerNorm(self.emb_dim)
        self.layer_normalizer_2 = nn.LayerNorm(self.emb_dim)
        self.linear_feed_forwarder = nn.Linear(self.emb_dim, self.emb_dim)
        


    def pool(self, x):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.cfg.model.pooling_type == 'cls': # type: ignore
            # cls is the first token of each sequence
            pooled_vec = x[:,0,:]
            
        elif self.cfg.model.pooling_type == 'max': # type: ignore
            pooled_vec = x.max(dim=1)

        elif self.cfg.model.pooling_type == 'mean': # type: ignore
            pooled_vec = x.mean(dim=1)


        return pooled_vec


    def encode_this_now(self, x):
        """
        Run a forward pass on the model
        """
        
        vectors_attended = self.multi_headed_attention.forward(x, x, x)
        
        vectors_residualized = x + vectors_attended
        
        vectors_layer_normed = self.layer_normalizer_1.forward(vectors_residualized)
        
        vectors_fed_forward = self.linear_feed_forwarder.forward(vectors_layer_normed)
        
        vectors_fed_and_residualized = vectors_fed_forward + vectors_layer_normed
        
        vectors_layer_normed_for_the_second_time = self.layer_normalizer_2.forward(vectors_fed_and_residualized)        

        return vectors_layer_normed_for_the_second_time
    
    
    def forward(self, x):
        token_fucker = self.tokenizer.encode(x, add_special_tokens=False, return_tensors="pt")[0]
        vectors = self.embeddings(token_fucker)
        ped_vectors = self.pe.forward(vectors)

        for _ in range(self.n_layers):
            ped_vectors = self.encode_this_now(ped_vectors)
        
        vectors_pooled = self.pool(ped_vectors) # batch_size x emb_dim
        
        return vectors_pooled
        
    def vectorize(self, x):
        """     
        x: [jeg hedder julian]
        token ids: [421, 583, 543]
        vectors: [v1, v2, v3]
        pe'd vectors: [v1 + pe1, v2 + pe2, v3 + pe3]
                
        returns:
            seq_len x embedding_dim (float32)
        """
        token_ids = self.tokenizer.encode(x, add_special_tokens=False, return_tensors="pt")[0]
        vectors = self.embeddings(token_ids)
        print(vectors.shape)
        ped_vectors = self.pe.forward(vectors)
        
        return ped_vectors


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -math.inf)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """A simple Multi-head attention layer."""
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None # store the attention maps
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) 

    def forward(self, x):
        x = x + self.pe[:, :x.size(0)] # type: ignore
        return self.dropout(x)
    
    
if __name__ == "__main__":
    test = ["the dog ate the cat", "the mouse sucked him off"]
    print(test)
    model = TransformerEncoder(2, 2, 768, 768, 1024)
    
    print(model.forward(test))