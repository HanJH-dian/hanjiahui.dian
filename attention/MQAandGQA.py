import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        #多头Query，单头Key/Value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim)  #输出维度为1个头
        self.v_proj = nn.Linear(embed_dim, self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, mask=None):
        B, L, _ = q.size()

        #分割q
        q = self.q_proj(q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]
        k = self.k_proj(k).unsqueeze(1)  # [B, 1, L, D]
        v = self.v_proj(v).unsqueeze(1)  # [B, 1, L, D]

        #计算注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, L, L]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, H, L, D]

        #合并输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.out_proj(attn_output), attn_weights


class GroupQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups):
        super().__init__()
        assert num_heads % num_groups == 0, "num_heads必须能被num_groups整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads

        #每个组共享一个Key/Value头
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim * num_groups)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * num_groups)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, mask=None):
        B, L, _ = q.size()
        group_size = self.num_heads // self.num_groups

        #Query投影：分割为多组
        q = self.q_proj(q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]

        #Key/Value投影：每组共享
        k = self.k_proj(k).view(B, L, self.num_groups, self.head_dim).transpose(1, 2)  # [B, G, L, D]
        v = self.v_proj(v).view(B, L, self.num_groups, self.head_dim).transpose(1, 2)  # [B, G, L, D]

        #重复组内Key/Value以匹配Query头数
        k = k.repeat_interleave(group_size, dim=1)  # [B, H, L, D]
        v = v.repeat_interleave(group_size, dim=1)  # [B, H, L, D]

        #计算注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, L, L]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, H, L, D]

        #合并输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.out_proj(attn_output), attn_weights


if __name__ == "__main__":
    #参数设置
    B, L, D = 2, 5, 64  # batch=2, seq_len=5, embed_dim=64
    H = 8  # 总头数
    G = 2  # GQA的组数

    #生成随机输入
    q = k = v = torch.randn(B, L, D)

    #初始化不同注意力机制
    mha = MultiQueryAttention(D, H)
    gqa = GroupQueryAttention(D, H, G)

    #计算注意力权重
    mha_output, mha_weights = mha(q, k, v)
    gqa_output, gqa_weights = gqa(q, k, v)

    #打印形状对比
    print("=== 注意力权重形状对比 ===")
    print(f"MHA权重形状: {mha_weights.shape}")  # [2, 8, 5, 5]
    print(f"GQA权重形状: {gqa_weights.shape}")  # [2, 8, 5, 5]（但内部计算共享KV）


    #打印KV Cache大小对比
    def get_kv_cache_size(module):
        if isinstance(module, MultiQueryAttention):
            k_size = (B, 1, L, D // H)
            v_size = (B, 1, L, D // H)
        elif isinstance(module, GroupQueryAttention):
            k_size = (B, G, L, D // H)
            v_size = (B, G, L, D // H)
        return (k_size, v_size)


    print("\n=== KV Cache空间对比 ===")
    print(f"MQA KV Cache大小: {get_kv_cache_size(mha)}")
    print(f"GQA KV Cache大小: {get_kv_cache_size(gqa)}")


#可视化对比
def plot_attention(weights, title):
    plt.matshow(weights[0, 0].detach().numpy())
    plt.title(title)
    plt.colorbar()
    plt.show()

plot_attention(mha_weights, "MQA Attention Weights")
plot_attention(gqa_weights, "GQA Attention Weights")
