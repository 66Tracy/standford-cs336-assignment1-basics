import torch.nn as nn
import torch
from einops import einsum

class customLinear(nn.Module):
    """
    Implement a nn.Linear based on torch.nn.Module
    """

    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        init_tensor = nn.init.trunc_normal_(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        ) # row vector to match the store
        self.W = nn.Parameter(init_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(self.W, x, "... d_out d_in, ... d_in -> ... d_out")
        print(y.shape)

if __name__ == "__main__":
    a = customLinear(728, 1064)
    x = torch.Tensor(728)
    y = a(x)


class customEmbedding(nn.Module):
    """
    Implement a nn.Embedding based on nn.Module
    """

    def __init__(self, 
                 num_embeddings:int,
                 embedding_dim:int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        init_tensor = nn.init.trunc_normal_(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        self.embeds = nn.Parameter(init_tensor)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_embeds = self.embeds[token_ids]
        return token_embeds


class customRMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: torch.device | None = None):
        super().__init__()
        init_tensor = nn.init.ones_(
            torch.empty(d_model, device=device, dtype=torch.float32)
        )
        self.g = nn.Parameter(init_tensor)
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input size : (batch_size, seq_len, d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        # 倒数平方根
        inv_rms = torch.rsqrt(mean_square + self.eps)
        # *是position-wise乘法
        result = (x * inv_rms) * self.g

        return result.to(in_dtype)



if __name__ == "__main__":
    e = customEmbedding(6, 5)
    token_ids = torch.tensor([1, 3, 5])
    print(f"token ids: {token_ids}")
    token_embeds = e(token_ids)
    print(f"embeddings: {e.embeds}")
    print(f"\n\nchosen: {token_embeds}")