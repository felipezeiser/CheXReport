from typing import Tuple
from torch import nn, Tensor
from torch.nn import MultiheadAttention

class DecoderLayer(nn.Module):
    """
    Camada de decodificação para um modelo de tradução ou geração de legendas, que incorpora atenção própria,
    atenção multi-cabeça e uma rede feedforward.
    
    Parâmetros:
        d_model (int): Dimensão das características de entrada/saída.
        num_heads (int): Número de cabeças para a atenção multi-cabeça.
        feedforward_dim (int): Dimensão interna da rede feedforward.
        dropout_rate (float): Probabilidade de dropout aplicada em várias operações.

    Métodos:
        forward(dec_inputs, enc_outputs, tgt_mask, tgt_pad_mask): Processa as entradas através das camadas do decodificador.
    """
    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int, dropout_rate: float):
        super(DecoderLayer, self).__init__()
        
        # Camada de atenção própria do decodificador
        self.self_attention = MultiheadAttention(d_model, num_heads, dropout=dropout_rate)
        self.self_attention_norm = nn.LayerNorm(d_model)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        
        # Camada de atenção multi-cabeça que interage com as saídas do codificador
        self.encoder_attention = MultiheadAttention(d_model, num_heads, dropout=dropout_rate)
        self.encoder_attention_norm = nn.LayerNorm(d_model)
        self.encoder_attention_dropout = nn.Dropout(dropout_rate)
        
        # Rede feedforward no decodificador
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_dim, d_model)
        )
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward_dropout = nn.Dropout(dropout_rate)

    def forward(self, dec_inputs: Tensor, enc_outputs: Tensor, tgt_mask: Tensor, tgt_pad_mask: Tensor) -> Tuple[Tensor, Tensor]:
        # Atualizações residuais com normalização e dropout para a atenção própria
        self_attn_output, _ = self.self_attention(
            dec_inputs, dec_inputs, dec_inputs, attn_mask=tgt_mask, key_padding_mask=tgt_pad_mask
        )
        dec_inputs = dec_inputs + self.self_attention_dropout(self_attn_output)
        self_attn_output = self.self_attention_norm(dec_inputs)

        # Atualizações residuais para a atenção com as saídas do codificador
        encoder_attn_output, attn_weights = self.encoder_attention(
            self_attn_output, enc_outputs, enc_outputs, average_attn_weights=False
        )
        self_attn_output = self_attn_output + self.encoder_attention_dropout(encoder_attn_output)
        encoder_attn_output = self.encoder_attention_norm(self_attn_output)

        # Rede feedforward
        ff_output = self.feedforward(encoder_attn_output)
        ff_output = encoder_attn_output + self.feedforward_dropout(ff_output)
        ff_output = self.feedforward_norm(ff_output)

        return ff_output, attn_weights


if __name__ == "__main__":
    import torch

    src_img = torch.rand(196, 10, 512)  # B, encode, embed
    captions = torch.rand(52, 10, 512)  # Max_len, B, embed_dim
    decoder_layer = DecoderLayer(512, 8, 2048, 0.1)
    output, attention_weights = decoder_layer(captions, src_img, None, None)
    print(output.size())  # Espera-se: torch.Size([52, 10, 512])
