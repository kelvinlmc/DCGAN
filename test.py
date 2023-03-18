import torch
from DCN import Generator
from  torchvision.utils import make_grid
import matplotlib.pyplot as plt

device='cuda' if torch.cuda.is_available() else 'cpu'
gen=Generator(100,1,64).to(device)
gen.load_state_dict(torch.load('./model/model_13900.pth',map_location='cpu'))
noise = torch.randn(128, 100, 1, 1).to(device)
fake=gen(noise)


def show_image(image_tensor,size=(1,64,64)):
    image=image_tensor.view(-1,*size)
    make_image=make_grid(image[:9],nrow=3)
    plt.imshow(make_image.permute(1,2,0))
    plt.show()

show_image(fake)


