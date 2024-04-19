from torch import nn, Tensor
from torch.nn import MultiheadAttention

class ConvolutionalFeedForward(nn.Module):
    """
    Implementação de uma Rede Feedforward Convolucional como parte de uma camada de codificação.
    Utiliza convoluções 1D para emular operações totalmente conectadas sobre os recursos codificados.
    
    Parâmetros:
        encode_size (int): Dimensão espacial das características codificadas.
        embed_dim (int): Dimensão dos recursos embutidos.
        feedforward_dim (int): Dimensão dos recursos na camada oculta.
        dropout_rate (float): Taxa de dropout para regularização.

    Métodos:
        forward(inputs): Processa o tensor de entrada e retorna um tensor de saída modificado.
    """
    def __init__(self, encode_size: int, embed_dim: int, feedforward_dim: int, dropout_rate: float):
        super(ConvolutionalFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=encode_size, out_channels=feedforward_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim, out_channels=encode_size, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs_permuted = inputs.permute(1, 0, 2)
        output = self.conv1(inputs_permuted)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.dropout(output)
        output = self.layer_norm(output.permute(1, 0, 2) + inputs)
        return output


class SelfAttention(nn.Module):
    """
    Camada de Autoatenção que utiliza MultiheadAttention para processar características de imagem.

    Parâmetros:
        img_embed_dim (int): Dimensão das características embutidas das imagens.
        num_heads (int): Número de cabeças no modelo MultiheadAttention.
        dropout_rate (float): Taxa de dropout para regularização.

    Métodos:
        forward(enc_inputs): Recebe entradas codificadas e retorna suas representações após a autoatenção.
    """
    def __init__(self, img_embed_dim: int, num_heads: int, dropout_rate: float):
        super(SelfAttention, self).__init__()
        self.multi_head_attn = MultiheadAttention(embed_dim=img_embed_dim, num_heads=num_heads, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(img_embed_dim)

    def forward(self, enc_inputs: Tensor) -> Tensor:
        enc_outputs, _ = self.multi_head_attn(enc_inputs, enc_inputs, enc_inputs)
        enc_outputs = self.layer_norm(enc_outputs + enc_inputs)
        return enc_outputs


class EncoderLayer(nn.Module):
    """
    Camada do Encoder que integra a Autoatenção e a Rede Feedforward Convolucional.

    Parâmetros:
        img_encode_size (int): Tamanho da codificação das imagens.
        img_embed_dim (int): Dimensão das características embutidas das imagens.
        feedforward_dim (int): Dimensão dos recursos na camada oculta do feedforward.
        num_heads (int): Número de cabeças no modelo MultiheadAttention.
        dropout_rate (float): Taxa de dropout para regularização.

    Métodos:
        forward(enc_inputs): Processa entradas através das camadas de autoatenção e feedforward.
    """
    def __init__(self, img_encode_size: int, img_embed_dim: int, feedforward_dim: int, num_heads: int, dropout_rate: float):
        super(EncoderLayer, self).__init__()
        self.self_attention = SelfAttention(img_embed_dim, num_heads, dropout_rate)
        self.feed_forward = ConvolutionalFeedForward(img_encode_size, img_embed_dim, feedforward_dim, dropout_rate)

    def forward(self, enc_inputs: Tensor) -> Tensor:
        enc_outputs = self.self_attention(enc_inputs)
        enc_outputs = self.feed_forward(enc_outputs)
        return enc_outputs


if __name__ == "__main__":
    import torch
    src_img = torch.rand(196, 10, 512)  # Simulando imagens com dimensão [encode_size^2, batch_size, embed_dim]
    model_test = EncoderLayer(196, 512, 2048, 8, 0.1)
    output = model_test(src_img)
    print(output.size())  # Saída esperada: torch.Size([196, 10, 512])
