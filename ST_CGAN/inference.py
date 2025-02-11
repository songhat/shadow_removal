from .utils.data_loader import ImageTransformOwn
from .models.ST_CGAN import Generator,Generator_S
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms
from collections import OrderedDict
from PIL import Image
import torch
import os

torch.manual_seed(44)
# choose your device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def fix_model_state_dict(state_dict):
    '''
    remove 'module.' of dataparallel
    '''
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


def unnormalize(x):
    x = x.transpose(1, 3)
    # mean, std
    x = x * torch.Tensor((0.5,)) + torch.Tensor((0.5,))
    x = x.transpose(1, 3)
    return x


def predict_single_image(G1, G2, img, size):
    # img = Image.open(path).convert('RGB')
    width, height = img.width, img.height
    mean = (0.5,)
    std = (0.5,)
    img_transform = ImageTransformOwn(size=size, mean=mean, std=std)
    img = img.resize((size, size), Image.LANCZOS)
    img = img_transform(img)
    img = torch.unsqueeze(img, dim=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    G1.to(device)
    G2.to(device)

    """use GPU in parallel"""
    if device == 'cuda':
        G1 = torch.nn.DataParallel(G1)
        G2 = torch.nn.DataParallel(G2)
        print("parallel mode")

    print("device:{}".format(device))

    G1.eval()
    G2.eval()

    with torch.no_grad():
        detected_shadow = G1(img.to(device))
        detected_shadow = detected_shadow.to(torch.device('cpu'))
        concat = torch.cat([img, detected_shadow], dim=1)
        shadow_removal_image = G2(concat.to(device))
        shadow_removal_image = shadow_removal_image.to(torch.device('cpu'))

        grid = make_grid(torch.cat([unnormalize(img),
                                    unnormalize(torch.cat([detected_shadow, detected_shadow, detected_shadow], dim=1)),
                                    unnormalize(shadow_removal_image)],
                                   dim=0))

        detected_shadow = transforms.ToPILImage(mode='L')(unnormalize(detected_shadow)[0, :, :, :])
        detected_shadow = detected_shadow.resize((width, height), Image.LANCZOS)

        shadow_removal_image = transforms.ToPILImage(mode='RGB')(unnormalize(shadow_removal_image)[0, :, :, :])
        shadow_removal_image = shadow_removal_image.resize((width, height), Image.LANCZOS)

    return shadow_removal_image

def predict(image, checkpoints, image_size=256):
    G1 = Generator(input_channels=3, output_channels=1)
    G2 = Generator(input_channels=4, output_channels=3)

    G1_weights = torch.load(checkpoints[0],weights_only=True)
    G1.load_state_dict(fix_model_state_dict(G1_weights),strict=False)

    G2_weights = torch.load(checkpoints[1],weights_only=True)
    G2.load_state_dict(fix_model_state_dict(G2_weights),strict=False)

    # predict_single_image(G1, G2, image_path, out_path, image_size)

    return predict_single_image(G1, G2, image, image_size)
    
