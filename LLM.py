import torch
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers.image_utils import load_image
import pandas as pd

processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")