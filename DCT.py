import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from DCN import Discriminator, Generator, initialize_weights
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

device='cuda' if torch.cuda.is_available() else 'cpu'
LEARING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCH = 10
FEATURES_DISC = 64
FEATURES_GEN = 64

def show_tensor_images(image_tensor,name, num_images=9, size=(1, 64,64)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=3)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig(fname=name, figsize=[10, 10])

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
    ])

data=datasets.MNIST(root='./data',download=False,train=True,transform=transforms)
dataloader=DataLoader(data,batch_size=BATCH_SIZE,shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARING_RATE, betas=(0.5, 0.999))
opt_disd = optim.Adam(disc.parameters(), lr=LEARING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

#gen.train()
#disc.train()
sum_D_loss=[]
sum_G_loss=[]
step = 0
for epoch in range(NUM_EPOCH):
    print("---------epoch{}---------".format(epoch))
    for x,(real, _ ) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
        fake = gen(noise)
        disc_real = disc(real)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach())
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_fake + loss_disc_real) / 2
        sum_D_loss.append(loss_disc)
        disc.zero_grad()
        loss_disc.backward()
        opt_disd.step()

        output = disc(fake)
        loss_gen = criterion(output, torch.ones_like(output))
        sum_G_loss.append(loss_gen)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        step += 1
        if step % 100 == 0 and step > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCH}] Batch {x}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f} "
            )
            show_tensor_images(fake, "./data/pic/p{}.png".format(step))
            torch.save(gen.state_dict(),'./model/model_{}.pth'.format(step))

            #with torch.no_grad():
                #fake = gen(fixed_noise)
                #img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True )
                #img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True )
                #writer_real.add_image("Real", img_grid_real, global_step=step)
                #writer_fake.add_image("Fake", img_grid_fake, global_step=step)
    #print("生成器迭代损失值{}".format(sum_G_loss))
    #print("鉴别器迭代损失值{}".format(sum_D_loss))

