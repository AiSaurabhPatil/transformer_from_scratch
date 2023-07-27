import torch 
import torch.nn as nn
import numpy as np

class Input_embedding(nn.Module):
    
    def __init__(self , d_model : int , vocab_size : int):
        super().__init__()
        self.d_model = d_model 
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size , d_model)
    

    def forward(self, x): 
        return self.embedding(x) * np.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int , seq_len : int , dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape ( seq_len , d_model )
        pe = torch.zeros(self.seq_len , self.d_model)

        # create a vector of shape ( seq_len , 1 )
        position = torch.arange(0 , self.seq_len , dtype=torch.float).unsqueeze(1)
        deno_term = torch.exp(torch.arange(0 , self.d_model , 2 , dtype=torch.float) * -(np.log(10000.0) / self.d_model))

        # Apply the sin to even position 
        pe[:, 0::2] = torch.sin(position * deno_term)

        # Apply the cos to odd position
        pe[:, 1::2] = torch.cos(position * deno_term)

        pe = pe.unsqueeze(0) # (1 , seq_len , d_model)

        """
        register_buffer() is a method that allows you to register a tensor
        as a buffer of a torch.nn.Module instance. Buffers are parameters that are 
        not updated during the training process, and they are typically used to store
        persistent state variables in a model.
        """
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[: , :x.shape[1] , : ]).requires_grad(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self , eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) ## Multipled
        self.beta =nn.Parameter(torch.zeros(1))  ## Added 

    
    def forward(self, x ):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    

class FeedForwardBlock(nn.Module):
    def __init__(self , d_model : int ,d_ff: int , dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model , d_ff)
        self.linear_2 = nn.Linear(d_ff , d_model)
    
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttention(nn.Module):
    def __init__(self , d_model :int , h : int , dropout: float ): 
        super().__init__()
        self.d_model = d_model
        self.h = h 
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % h == 0
        self.d_k = d_model // h  ## return a whole no as output !! 

        self.w_q = nn.Linear(d_model , d_model) # query 
        self.w_k = nn.Linear(d_model , d_model) # key
        self.w_v = nn.Linear(d_model , d_model) # value

        self.w_o = nn.Linear(d_model , d_model) # output



    @staticmethod ## this allowes to use this function without creating a new object
    def attention( query , key, value , mask , dropout : nn.Dropout ):
        d_k = query.shape[-1]

        # ( batch , seq_len , d_model ) ----> ( batch ,h,seq_len , seq_len) 
        attention_scores = (query @key.transpose(-2 , -1)) / np.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_score = torch.softmax(attention_scores , dim = -1) # ( batch ,h, seq_len , seq_len )
        if dropout is not None :
            attention_score = dropout(attention_score)
        return (attention_score @ value) , attention_score
    

    def forward(self ,q,v,k ,mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        #(Batch , seq_len , d_model) ----> (Batch , seq_len , h , d_k) -----> (Batch , h , seq_len , d_k)
        query = query.view(query.shape[0], query.shape[1], self.h , self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h , self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h , self.d_k).transpose(1,2)
    
        x, self.attention_score = self.attention(query , key , value , mask , self.dropout)

        x = x.transpose(1 , 2).contiguous().view(x.shape[0], -1 , self.h * self.d_k)

        return self.w_o(x)
      

class ResidualConnection(nn.Module):
    def __init__(self , dropout: float):
        super().__init__()
        self.dropout = dropout
        self.norm = LayerNormalization()

    def forward(self, x,sublayer):
        return self.dropout(sublayer(self.norm(x)))
        

class EncoderBlock(nn.Module):
    def __init__(self , self_attention_block : MultiHeadAttention , feed_forward_block : FeedForwardBlock , dropout : float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self , x , src_mask):
        x = self.residual_connection[0](x , lambda:x self.self_attention_block(x , x , x , src_mask))
        x = self.residual_connection[1](x , self.feed_forward_block)
        return x
    
    

class Encoder(nn.Module):
    def __init__(self , layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norms = LayerNormalization()
    
    def forward(self ,x , mask):
        for layer in self.layers:
            x = layer(x , mask)
        return self.norms(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttention , 
                 cross_attention_block : MultiHeadAttention , feed_forward_block : FeedForwardBlock , dropout : float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self , x , encoder_output , src_mask , tgt_mask):
        x = self.residual_connection[0](x , lambda:x self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connection[1](x , lambda:x self.cross_attention_block(x , encoder_output,encoder_output,src_mask))
        x = self.residual_connection[2](x , self.feed_forward_block)    
        return x 
    

class Decoder(nn.Module):
    def __init__(self , layers : nn.ModuleList):
        super().__init__()
        self.layers = layers 
        self.norm = LayerNormalization()

    
    def forward(self, x , encoder_output , src_mask , tgt_mask):
        for layer in self.layers:
            x = layer(x ,encoder_output , src_mask , tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int , vocab_size : int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model , vocab_size)
    
    def forward(self , x):
        return torch.log_softmax(self.proj(x) , dim=-1)
    


class Transformer(nn.Module):
    def __init__(self,encoder:Encoder , decoder : Decoder ,src_embed : Input_embedding 
                 ,tgt_embed : Input_embedding , src_pos : PositionalEncoding , tgt_pos : PositionalEncoding,
                 projection_layer : ProjectionLayer): 
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder 
        self.src_embed = src_embed
        self.tgt_emb = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    
    def encode(self,src , src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src , src_mask)
    

    def decode(self, encoder_output , src_mask , tgt , tgt_mask ):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt , encoder_output , src_mask , tgt_mask)

    def projection(self, x):
        return self.projection_layer(x)
    
