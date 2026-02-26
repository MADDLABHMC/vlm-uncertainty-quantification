"""
CLIPSeg model with encoder dropout for MC Dropout inference.
"""
import torch.nn as nn
from transformers import AutoProcessor, CLIPSegForImageSegmentation


class CLIPSegWithEncoderDropout(nn.Module):
    """
    Wrapper around CLIPSeg that adds dropout to the vision encoder.

    We modify the CLIP vision model itself to have dropout,
    then use the standard CLIPSeg forward pass.
    """

    def __init__(self, base_model, dropout_rate=0.2):
        super().__init__()
        self.model = base_model
        self.dropout_rate = dropout_rate

        # Add dropout layers to the vision encoder's MLP layers
        # These are the feedforward layers after each attention block
        self._add_dropout_to_encoder()

    def _add_dropout_to_encoder(self):
        """
        Add dropout to the vision encoder's MLP layers.
        Each transformer block has: Attention -> LayerNorm -> MLP -> LayerNorm
        We add dropout after the MLP.
        """
        encoder_layers = self.model.clip.vision_model.encoder.layers

        for layer in encoder_layers:
            original_mlp_forward = layer.mlp.forward
            dropout = nn.Dropout(p=self.dropout_rate)

            def make_mlp_forward_with_dropout(orig_forward, dropout_layer):
                def forward_with_dropout(hidden_states):
                    output = orig_forward(hidden_states)
                    if dropout_layer.training:
                        output = dropout_layer(output)
                    return output

                return forward_with_dropout

            layer.mlp.forward = make_mlp_forward_with_dropout(
                original_mlp_forward, dropout
            )

            if not hasattr(layer.mlp, "dropout_layers"):
                layer.mlp.dropout_layers = []
            layer.mlp.dropout_layers.append(dropout)

    def enable_dropout(self):
        """Enable dropout in the encoder for MC sampling."""
        for layer in self.model.clip.vision_model.encoder.layers:
            if hasattr(layer.mlp, "dropout_layers"):
                for dropout in layer.mlp.dropout_layers:
                    dropout.train()

        for module in self.model.clip.vision_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def disable_dropout(self):
        """Disable dropout for deterministic inference."""
        for layer in self.model.clip.vision_model.encoder.layers:
            if hasattr(layer.mlp, "dropout_layers"):
                for dropout in layer.mlp.dropout_layers:
                    dropout.eval()

        for module in self.model.clip.vision_model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    def forward(self, **kwargs):
        """Standard forward pass - dropout is built into the encoder now."""
        return self.model(**kwargs)


def load_model(
    model_name: str = "CIDAS/clipseg-rd64-refined",
    dropout_rate: float = 0.1,
):
    """
    Load CLIPSeg model with encoder dropout.

    Args:
        model_name: HuggingFace model identifier
        dropout_rate: Dropout probability in encoder (default: 0.1)

    Returns:
        model: CLIPSegWithEncoderDropout
        processor: AutoProcessor
    """
    processor = AutoProcessor.from_pretrained(model_name)
    base_model = CLIPSegForImageSegmentation.from_pretrained(model_name)
    model = CLIPSegWithEncoderDropout(base_model, dropout_rate=dropout_rate)
    model.eval()
    return model, processor
