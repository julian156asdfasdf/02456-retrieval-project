import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int, pooling_type: str, batch_size=32):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.batch_size = batch_size
        self.pooling_type = pooling_type

        self.setup_model()


    def setup_model(self):
        """
        Instantiate the model.
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
        self.embeddings = nn.Embedding(self.tokenizer.vocab_size, self.emb_dim)
        self.pe = PositionalEncoding(self.emb_dim, 0.1)

        self.layers = []
        for _ in range(self.n_layers):
            self.layers.append(EncoderLayer(n_heads = self.n_heads, emb_dim = self.emb_dim, feed_forward_dim = self.feed_forward_dim, dropout = 0.2))
        self.layers = nn.ModuleList(self.layers)


    def pool(self, x, mask = None):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        
        mask = mask.unsqueeze(-1)

        if self.pooling_type == 'cls':
            # cls is the first token of each sequence, now this works since we added special tokens in the encoder
            pooled_vec = x[:,0,:]
            
        elif self.pooling_type == 'max': 
            x_masked = x.masked_fill(mask == 0, -1000) # We make the masked tokens really small, these are the embeddings where the mask is 0.
            pooled_vec = x_masked.max(dim=1).values

        elif self.pooling_type == 'mean':
            # We do not want for the masked tokens to count toward the mean, so those become zero, then we dvide by sum of mask as oonly tokens that are not padding tokens should count.
            x_masked = (x * mask) 
            pooled_vec = x_masked.sum(dim = 1)/mask.sum(dim=1)


        return pooled_vec


    
    
    def forward(self, x):
        token_dict = self.tokenizer(x, add_special_tokens=True, padding = True, return_tensors="pt", truncation = True, max_length = 512)
        tokens = token_dict["input_ids"].to(self.device)
        mask = token_dict["attention_mask"].to(self.device)

        vectors = self.embeddings(tokens)
        ped_vectors = self.pe.forward(vectors)

        for layer in self.layers:
            ped_vectors = layer(ped_vectors, mask.unsqueeze(1))
        
        vectors_pooled = self.pool(ped_vectors, mask) # batch_size x emb_dim
        
        return vectors_pooled


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, emb_dim, feed_forward_dim, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(n_heads, emb_dim)
        self.layernorm1 = nn.LayerNorm(emb_dim)
        self.layernorm2 = nn.LayerNorm(emb_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(emb_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, emb_dim),
        )

    def forward(self, x, mask=None):
        x_attendend = self.attention.forward(x, x, x, mask)
        x = self.layernorm1(x + x_attendend)
        x_feed_forward = self.feedforward(x)
        x = self.layernorm2(x + x_feed_forward)
        return x    



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
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
    
    
if __name__ == "__main__":
    test = ["the dog ate the cat", "the mouse sucked him off and looked at the sun"]
    print(test)
    model = TransformerEncoder(n_layers = 6, n_heads = 4, emb_dim = 768,hidden_dim = 768,feed_forward_dim =  1024, pooling_type = 'max')
    
    print(model.forward(test).shape)