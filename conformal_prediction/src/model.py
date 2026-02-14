"""
CLIPSeg model wrapper for semantic segmentation.
"""
import torch
from transformers import AutoProcessor, CLIPSegForImageSegmentation


class CLIPSegModel:
    """Wrapper for CLIPSeg segmentation model."""
    
    def __init__(self, model_name: str = "CIDAS/clipseg-rd64-refined"):
        """Initialize CLIPSeg model and processor.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name)
        self.model.eval()
    
    def predict(self, image, texts: list[str]) -> torch.Tensor:
        """Run segmentation prediction.
        
        Args:
            image: PIL Image or path to image
            texts: List of text prompts for segmentation classes
            
        Returns:
            Probability tensor of shape (H, W, num_classes)
        """
        inputs = self.processor(
            text=texts, 
            images=[image] * len(texts), 
            padding=True, 
            return_tensors="pt"
        )
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
        
        logits = outputs.logits  # Shape: (num_classes, H, W)
        probs = torch.sigmoid(logits.permute(1, 2, 0))  # Shape: (H, W, num_classes)
        
        return probs