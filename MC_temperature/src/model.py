"""
CLIPSeg model with dropout before the final decoder layer for Temperature-Scaled MC Dropout.

Per the paper: Insert dropout (p=0.3) before the final linear layer. Train normally.
At test time: MC dropout with N forward passes, apply temperature T to logits before softmax,
average the softmax outputs, and optimize T on a held-out validation set.
"""
import torch.nn as nn
from transformers import AutoProcessor, CLIPSegForImageSegmentation


class CLIPSegWithDecoderDropout(nn.Module):
    """
    Wrapper around CLIPSeg that adds dropout (p=0.3) before the final decoder layer.

    The dropout is inserted before the transposed_convolution in the decoder,
    matching the paper's "dropout before the final linear layer" for dense prediction.
    """

    def __init__(self, base_model, dropout_rate: float = 0.3):
        super().__init__()
        self.model = base_model
        self.dropout_rate = dropout_rate
        self._decoder_dropout = nn.Dropout(p=dropout_rate)
        self._patch_decoder()

    def _patch_decoder(self):
        """Insert dropout before the decoder's final transposed_convolution layer."""
        decoder = self.model.decoder
        original_conv = decoder.transposed_convolution
        # Wrap: dropout -> conv (dropout applied to decoder features before final conv)
        decoder.transposed_convolution = nn.Sequential(
            self._decoder_dropout,
            original_conv,
        )

    def enable_dropout(self):
        """Enable dropout for MC sampling at test time."""
        self._decoder_dropout.train()

    def disable_dropout(self):
        """Disable dropout for deterministic inference."""
        self._decoder_dropout.eval()

    def forward(self, **kwargs):
        """Standard forward pass."""
        return self.model(**kwargs)


def load_model(
    model_name: str = "CIDAS/clipseg-rd64-refined",
    dropout_rate: float = 0.3,
):
    """
    Load CLIPSeg model with decoder dropout for Temperature-Scaled MC Dropout.

    Args:
        model_name: HuggingFace model identifier
        dropout_rate: Dropout probability before final layer (default: 0.3 per paper)

    Returns:
        model: CLIPSegWithDecoderDropout
        processor: AutoProcessor
    """
    processor = AutoProcessor.from_pretrained(model_name)
    base_model = CLIPSegForImageSegmentation.from_pretrained(model_name)
    model = CLIPSegWithDecoderDropout(base_model, dropout_rate=dropout_rate)
    model.eval()
    return model, processor
