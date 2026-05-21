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

out = scaled_dot_product(Q, K, V)
print(out.shape)


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
        
        
        
mha = MultiHeadAttention(d_model=8, num_heads=2)
Q = torch.randn(4, 8)
K = torch.randn(4, 8)
V = torch.randn(4, 8)
out = mha(Q, K, V)
print(out.shape) 