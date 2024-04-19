from torch import nn, Tensor
import timm

class ImageFeatureExtractor(nn.Module):
    """
    Módulo para extrair características de imagens usando uma rede neural pré-treinada.
    O módulo utiliza um Swin Transformer para processar imagens e produzir um tensor de características.

    Parâmetros:
        encode_size (int): O tamanho final da dimensão espacial do tensor de características.
        embed_dim (int): O número de canais desejado no tensor de características após o downsampling.

    Métodos:
        forward(images): Recebe um batch de imagens e retorna um tensor de características correspondente.
        fine_tune(enable): Ativa ou desativa o fine-tuning nos parâmetros do Swin Transformer.
    """
    def __init__(self, encode_size=14, embed_dim=512):
        super(ImageFeatureExtractor, self).__init__()

        self.embedding_dimension = embed_dim
        # Inicializa um Swin Transformer pré-treinado
        self.swin_transformer = timm.create_model('swin_large_patch4_window12_384', pretrained=True, features_only=True)
        
        # Downsampling para ajustar as dimensões dos canais
        self.channel_adjust = nn.Conv2d(
            in_channels=self.swin_transformer.feature_info.channels()[-1],
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu_activation = nn.ReLU(inplace=True)
        
        # Redimensionamento adaptativo para a dimensão espacial desejada
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(encode_size)

    def forward(self, images: Tensor) -> Tensor:
        batch_size = images.size()[0]

        # Processamento pela Swin Transformer
        feature_maps = self.swin_transformer(images)
        last_feature_map = feature_maps[-1]  # Assume-se que o último mapa de características é o desejado

        # Ajuste de dimensões, se necessário
        if last_feature_map.shape[1] != self.swin_transformer.feature_info.channels()[-1]:
            last_feature_map = last_feature_map.permute(0, 3, 1, 2)

        # Downsampling para dimensionar os canais
        processed_features = self.relu_activation(self.batch_norm(self.channel_adjust(last_feature_map)))

        # Redimensionamento para as dimensões finais
        resized_features = self.adaptive_pooling(processed_features)
        output_features = resized_features.view(batch_size, self.embedding_dimension, -1).permute(0, 2, 1)
        
        return output_features

    def fine_tune(self, enable: bool = True):
        """
        Ativa ou desativa o treinamento dos parâmetros do Swin Transformer.

        Parâmetros:
            enable (bool): Define se o fine-tuning deve ser ativado ou não.
        """
        for param in self.swin_transformer.parameters():
            param.requires_grad = enable
