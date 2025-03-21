import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, use_kv_cache=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        #初始化Q、K、V的线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        #KV Cache相关
        self.use_kv_cache = use_kv_cache
        self.k_cache = None
        self.v_cache = None

    #将输入张量分割为多头
    def _split_heads(self, x):
        batch_size, seq_len, embed_dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  #顺序重新排列（样本数量，头数量，样本元素数量，头维度）

    #将多头合并
    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        return x.contiguous().view(x.size(0), x.size(1), self.embed_dim)

    def forward(self, q, k, v, mask=None):
        #生成新Q、K、V
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        #处理KV Cache
        if self.use_kv_cache and self.k_cache is not None:
            k = torch.cat([self.k_cache, k], dim=1)
            v = torch.cat([self.v_cache, v], dim=1)
        if self.use_kv_cache:
            self.k_cache = k
            self.v_cache = v

        #分割多头
        q = self._split_heads(q)  #[batch, num_heads, q_len, head_dim]
        k = self._split_heads(k)
        v = self._split_heads(v)

        #公式计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        if mask is not None:#应用掩码（如果有）
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)#softmax归一化
        attn_output = torch.matmul(attn_weights, v)#加权求和

        ttn_output = self._merge_heads(attn_output)#合并多头并返回
        return self.out_proj(attn_output), attn_weights

    #重置KV缓存
    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None


if __name__ == "__main__":
    #参数设置
    batch_size = 2
    seq_len = 5
    embed_dim = 64
    num_heads = 4

    q = k = v = torch.randn(batch_size, seq_len, embed_dim)#生成随机输入
    mha = MultiHeadAttention(embed_dim, num_heads)#初始化
    attn_output, attn_weights = mha(q, k, v)#计算注意力

    #输出验证
    print("输入形状:", q.shape)  #[2, 5, 64]
    print("注意力权重形状:", attn_weights.shape)  #[2, 4, 5, 5]
    print("输出形状:", attn_output.shape)  #[2, 5, 64]

    #可视化第一个样本第一个头的注意力权重
    import matplotlib.pyplot as plt

    plt.matshow(attn_weights[0, 0].detach().numpy())
    plt.title("Attention Weights (Head 1)")
    plt.show()
