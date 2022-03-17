import PIL
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from PIL import Image
import cv2
import IPython
from IPython.display import Image

from models import MainModel
from utils import lab_to_rgb

if __name__ == '__main__':
    model = MainModel()
    
    # You first need to download the final_model_weights.pt file from my drive
    # using the command: gdown --id 1lR6DcS4m5InSbZ5y59zkH2mHt_4RQ2KV
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load(
            "final_model_weights.pt",
        )
    )
     path = "/content/img_9.jfif"
    img = PIL.Image.open(path)
    img = img.resize((256, 256))
    # to make it between -1 and 1
    img = transforms.ToTensor()(img)[:1] * 2. - 1.
    model.eval()
    with torch.no_grad():
        preds = model.net_G(img.unsqueeze(0).to(device))
    colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
    colorize_image(path)
