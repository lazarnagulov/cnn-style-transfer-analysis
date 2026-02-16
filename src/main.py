import os
import sys
import torch

from model import run_style_transfer
from utils import load_image, save_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def main(argv) -> None:
    assert len(argv) - 1 == 2, f"Expected style and content images, found { len(argv) } image(s)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    imsize = 512 if torch.cuda.is_available() else 256
    
    style_img   = load_image(argv[1], imsize, device)
    content_img = load_image(argv[2], imsize, device)
    input_img   = content_img.clone()

    output = run_style_transfer(content_img, style_img, input_img)
    save_image(output, RESULTS_DIR)

if __name__ == "__main__":
    main(sys.argv)