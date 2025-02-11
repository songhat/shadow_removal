from ST_CGAN.utils.data_loader import ImageTransformOwn
from ST_CGAN.models.ST_CGAN import Generator,Generator_S
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


def predict_single_image(G1, G2, path, out_path, size):
    img = Image.open(path).convert('RGB')
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

        save_image(grid, out_path + '/grid_' + path.split('/')[-1])

        detected_shadow = transforms.ToPILImage(mode='L')(unnormalize(detected_shadow)[0, :, :, :])
        detected_shadow = detected_shadow.resize((width, height), Image.LANCZOS)
        detected_shadow.save(out_path + '/detected_shadow_' + path.split('/')[-1])

        shadow_removal_image = transforms.ToPILImage(mode='RGB')(unnormalize(shadow_removal_image)[0, :, :, :])
        shadow_removal_image = shadow_removal_image.resize((width, height), Image.LANCZOS)
        shadow_removal_image.save(out_path + '/shadow_removal_image_' + path.split('/')[-1])


def main(image_dir, out_path, checkpoint_num, image_size):
    G1 = Generator(input_channels=3, output_channels=1)
    G2 = Generator(input_channels=4, output_channels=3)

    '''load'''
    if checkpoint_num is not None:
        print('load checkpoint ' + checkpoint_num)

        G1_weights = torch.load('./checkpoints0/ST-CGAN_G1_' + checkpoint_num + '.pth')
        G1.load_state_dict(fix_model_state_dict(G1_weights),strict=False)

        G2_weights = torch.load('./checkpoints0/ST-CGAN_G2_' + checkpoint_num + '.pth')
        G2.load_state_dict(fix_model_state_dict(G2_weights),strict=False)

    # predict_single_image(G1, G2, image_path, out_path, image_size)

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            # 为每张图片生成对应的输出文件名
            predict_single_image(G1, G2, image_path, out_path, image_size)


if __name__ == "__main__":
    image_dir= './imgs2'  # 替换为实际的图像路径
    out_path = './test_result'
    checkpoint_num = '250'
    image_size = 256
    main(image_dir, out_path, checkpoint_num, image_size)