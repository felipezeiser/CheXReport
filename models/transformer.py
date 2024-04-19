from copy import deepcopy
from typing import Tuple

import math
import torch
from torch import nn, Tensor

from .encoder import EncoderLayer
from .decoder import DecoderLayer

class PositionalEncoding(nn.Module):
    """
    Adiciona codificação posicional à entrada do modelo para incluir informações sequenciais.
    
    Parâmetros:
        d_model (int): Dimensão das características do modelo.
        max_len (int): Comprimento máximo esperado para as entradas.
        
    Método:
        forward(x): Adiciona codificação posicional ao tensor de entrada.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Criar codificação posicional
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Registra a codificação posicional como buffer não treinável
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x * math.sqrt(self.d_model)
        x += self.pe[:, :x.size(1)]
        return x

class Encoder(nn.Module):
    """
    Módulo Encoder do Transformer composto por várias camadas idênticas de EncoderLayer.
    
    Parâmetros:
        layer (EncoderLayer): Uma instância de EncoderLayer para ser copiada.
        num_layers (int): Número de camadas de codificação.
        
    Métodos:
        forward(x): Processa o tensor de entrada através das camadas de codificação.
    """
    def __init__(self, layer: EncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    """
    Módulo Decoder do Transformer para processar a saída do encoder e gerar sequências de texto.
    
    Parâmetros:
        layer (DecoderLayer): Uma instância de DecoderLayer para ser copiada.
        vocab_size (int): Tamanho do vocabulário.
        d_model (int): Dimensão das características do modelo.
        num_layers (int): Número de camadas do decodificador.
        max_len (int): Comprimento máximo das sequências de texto.
        dropout (float): Taxa de dropout.
        pad_id (int): ID do token de preenchimento.
        
    Métodos:
        forward(tgt_cptn, src_img): Processa as entradas do decodificador e gera previsões e pesos de atenção.
    """
    def __init__(self, layer: DecoderLayer, vocab_size: int, d_model: int, num_layers: int, max_len: int, dropout: float, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.caption_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=dropout)

    def get_attn_subsequent_mask(self, size: int) -> Tensor:
        return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)

    def forward(self, tgt_cptn: Tensor, src_img: Tensor) -> Tuple[Tensor, Tensor]:
        tgt_pad_mask = (tgt_cptn == self.pad_id)
        tgt_mask = self.get_attn_subsequent_mask(tgt_cptn.size()[1])
        tgt_mask = tgt_mask.to(tgt_cptn.device)

        tgt_cptn = self.caption_embedding(tgt_cptn).permute(1, 0, 2)
        tgt_cptn = self.positional_encoding(tgt_cptn)
        tgt_cptn = self.dropout(tgt_cptn)

        attns_all = []
        for layer in self.layers:
            tgt_cptn, attns = layer(tgt_cptn, src_img, tgt_mask, tgt_pad_mask)
            attns_all.append(attns)
        attns_all = torch.stack(attns_all)

        return tgt_cptn, attns_all

class Transformer(nn.Module):
    """
    Implementação completa do modelo Transformer para tarefas de processamento de linguagem natural, como tradução ou geração de texto baseada em imagem.
    
    Parâmetros:
        vocab_size (int): Tamanho do vocabulário.
        d_model (int): Dimensão das características do modelo.
        img_encode_size (int): Tamanho da codificação das imagens.
        enc_ff_dim (int), dec_ff_dim (int): Dimensões internas das redes feedforward dos módulos de codificação e decodificação.
        enc_n_layers (int), dec_n_layers (int): Números de camadas nos módulos de codificação e decodificação.
        enc_n_heads (int), dec_n_heads (int): Números de cabeças de atenção nos módulos de codificação e decodificação.
        max_len (int): Comprimento máximo das sequências de texto.
        dropout (float): Taxa de dropout.
        pad_id (int): ID do token de preenchimento.
        
    Métodos:
        forward(images, captions): Processa imagens e legendas, produzindo saídas preditivas e pesos de atenção.
    """
    def __init__(self, vocab_size: int, d_model: int, img_encode_size: int, enc_ff_dim: int, dec_ff_dim: int, enc_n_layers: int, dec_n_layers: int, enc_n_heads: int, dec_n_heads: int, max_len: int, dropout: float = 0.1, pad_id: int = 0):
        super(Transformer, self).__init__()
        encoder_layer = EncoderLayer(img_encode_size=img_encode_size, img_embed_dim=d_model, feedforward_dim=enc_ff_dim, num_heads=enc_n_heads, dropout_rate=dropout)
        decoder_layer = DecoderLayer(d_model=d_model, num_heads=dec_n_heads, feedforward_dim=dec_ff_dim, dropout_rate=dropout)
        self.encoder = Encoder(layer=encoder_layer, num_layers=enc_n_layers)
        self.decoder = Decoder(layer=decoder_layer, vocab_size=vocab_size, d_model=d_model, num_layers=dec_n_layers, max_len=max_len, dropout=dropout, pad_id=pad_id)
        self.predictor = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, images: Tensor, captions: Tensor) -> Tuple[Tensor, Tensor]:
        images_encoded = self.encoder(images.permute(1, 0, 2))
        captions_output, attns = self.decoder(captions, images_encoded)
        predictions = self.predictor(captions_output).permute(1, 0, 2)
        return predictions.contiguous(), attns.contiguous()

if __name__ == "__main__":
    src_img = torch.rand(10, 196, 512)
    captions = torch.randint(0, 52, (10, 30), dtype=torch.long)
    model = Transformer(52, 512, 196, 512, 2048, 2, 8, 8, 8, 30, 0.1, 0)
    output, attn_weights = model(src_img, captions)
    print(output.size(), attn_weights.size())
