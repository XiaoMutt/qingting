import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
COLLECTED_BACKGROUND_IMAGE_SIZE = (80, 80)
COLLECTED_DIGIT_IMAGE_SIZE = (16, 16)
