from math import sqrt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TimeEmbedding(nn.Module):
    """
    时间嵌入模块
    """
    def __init__(self, embed_dim=2):
        """
        初始化时间嵌入层
        :param embed_dim: 每个离散时间特征的嵌入维度
        """
        super(TimeEmbedding, self).__init__()
        self.holiday_embedding = nn.Embedding(3, embed_dim)  # 节假日嵌入
        self.month_embedding = nn.Embedding(12, embed_dim)   # 月份嵌入
        # self.week_embedding = nn.Embedding(7, embed_dim)   # 星期嵌入
        self.hour_embedding = nn.Embedding(24, embed_dim)  # 小时嵌入

    def forward(self, x):
        """
        前向传播
        :param time_features: [batch_size, seq_len, 4] 包含节假日、月份、星期、小时的离散特征
        :return: [batch_size, seq_len, 4 * embed_dim] 时间嵌入
        """
        holiday_emb = self.holiday_embedding(x[:, :, -3].long())  # [batch_size, seq_len, embed_dim]
        month_emb = self.month_embedding(x[:, :, -2].long())      # [batch_size, seq_len, embed_dim]
        hour_emb = self.hour_embedding(x[:, :, -1].long())       # [batch_size, seq_len, embed_dim]

        # 拼接所有时间嵌入
        time_emb = torch.cat([x[:, :, :-3],holiday_emb, month_emb, hour_emb], dim=-1)  # [batch_size, seq_len, 4 * embed_dim]
        return time_emb

class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars

class ExEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1, time_embed = 2):
        super(ExEmbedding, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.TimeEmbedding = TimeEmbedding(embed_dim=time_embed)
        
    def forward(self, x):
        x = self.TimeEmbedding(x)
        x = x.permute(0, 2, 1)
        # x: [Batch, Variate, Seq_len]
        x = self.value_embedding(x)
        # x: [Batch, Variate, d_model]
        return self.dropout(x)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,):
        super(AttentionLayer, self).__init__()

        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class FullAttention(nn.Module):
    def __init__(self, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross):
        for layer in self.layers:
            x = layer(x, cross)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(x, x, x)[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(x_glb, cross, cross)[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class TimePower(nn.Module):
    """
    For Short-term Load Forecasting
    """
    def __init__(self,seq_len, pre_len, d_model, patch_len, n_heads, d_ff, e_layers,use_norm=False, time_embed=2):
        super(TimePower, self).__init__()
        self.pre_len = pre_len
        self.time_embed = time_embed
        self.en_embedding = EnEmbedding(n_vars=1, d_model=d_model, patch_len=patch_len, dropout=0.1)
        self.ex_embedding = ExEmbedding(seq_len=seq_len+pre_len, d_model=d_model, dropout=0.1,time_embed = self.time_embed)
        self.patch_num = int(seq_len//patch_len)
        self.head_nf = d_model*(self.patch_num + 1)
        self.use_norm = use_norm
        self.encoder = Encoder(
            [
                EncoderLayer(
                    self_attention=AttentionLayer(FullAttention(attention_dropout=0.1,output_attention=False),
                        d_model=d_model, n_heads=8),
                    cross_attention=AttentionLayer(FullAttention(attention_dropout=0.1,output_attention=False),
                        d_model=d_model, n_heads=n_heads),
                    d_model=d_model, d_ff=d_ff, dropout=0.1, activation="relu",
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.head = FlattenHead(n_vars=1, nf=self.head_nf, target_window=pre_len, head_dropout=0)
    def forward(self, En_X, Ex_X):
        if self.use_norm:
            # Normalization(from Non-stationary Transformer)
            En_X_means = En_X.mean(1, keepdim=True).detach()
            Ex_X_means = Ex_X[:, :, :-4].mean(1, keepdim=True).detach()
            
            En_X = En_X - En_X_means
            Ex_X[:, :, :-4] = Ex_X[:, :, :-4] - Ex_X_means
            
            En_X_stdev = torch.sqrt(torch.var(En_X, dim=1, keepdim=True, unbiased=False) + 1e-5)
            En_X /= En_X_stdev
            Ex_X_stdev = torch.sqrt(torch.var(Ex_X[:, :, :-4], dim=1, keepdim=True, unbiased=False) + 1e-5)
            Ex_X[:, :, :-4] /= Ex_X_stdev
            
        en_embed, n_vars = self.en_embedding(En_X.permute(0, 2, 1))
        ex_embed = self.ex_embedding(Ex_X)
        
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # After being permuted, [batch, n_vars, d_model, patch_num + 1]
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)
        
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out*(En_X_stdev.repeat(1, self.pre_len, 1))
            dec_out += dec_out + En_X_means.repeat(1, self.pre_len, 1)
        return dec_out