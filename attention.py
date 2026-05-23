import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product(Q,K,V):
    d_k=Q.shape[-1]
    scores=F.softmax(Q@K.transpose(-2,-1)/d_k**0.5,dim=-1)@V
    return scores

Q = torch.randn(4, 8)  # 4 tokens, d_k=8
K = torch.randn(4, 8)
V = torch.randn(4, 8)



class MultiHeadAttention(nn.Module):
    def __init__(self,d_model, num_heads):
        super().__init__()
        
        # d_model must be divisible by num_heads
        
        self.d_k = d_model//num_heads
        self.num_heads=num_heads
        
        # define: d_k, W_q, W_k, W_v, W_o (output projection)
        
        self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.Linear(d_model,d_model)
        self.W_o=nn.Linear(d_model,d_model)
        
    def forward(self, Q, K, V):
        
        # 1. Project Q, K, V through their weight matrices
        
        Q=self.W_q(Q)
        K=self.W_k(K)
        V=self.W_v(V)
        seq_len = Q.shape[0]
        
        # 2. Split into num_heads
        
        Q=Q.view(seq_len,self.num_heads,self.d_k).transpose(0,1)
        K=K.view(seq_len,self.num_heads,self.d_k).transpose(0,1)
        V=V.view(seq_len,self.num_heads,self.d_k).transpose(0,1)
        
        # 3. Run scaled_dot_product_attention on each head
        
        score_attended= scaled_dot_product(Q,K,V)
        
        # 4. Concatenate head
        
        out = score_attended.transpose(0,1).contiguous().view(seq_len,-1)
        
        # 5. Project through W_o
        
        out = self.W_o(out)
        return out
        
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn1 = nn.Linear(d_model, ffn_hidden)   # expand
        self.ffn2 = nn.Linear(ffn_hidden, d_model)   # compress
        self.norm1 = nn.LayerNorm(d_model)            # after attention
        self.norm2 = nn.LayerNorm(d_model)            # after FFN

    def forward(self, x):
        # sub-layer 1: attention + residual + norm
        attended = self.attention(x, x, x)
        x = self.norm1(x + attended)
        
        # sub-layer 2: FFN + residual + norm
        ffn_out = torch.relu(self.ffn1(x))
        ffn_out = self.ffn2(ffn_out)
        x = self.norm2(x + ffn_out)
        
        return x
        
block = EncoderBlock(d_model=8, num_heads=2, ffn_hidden=32)
x = torch.randn(4, 8)
out = block(x)
print(out.shape)  