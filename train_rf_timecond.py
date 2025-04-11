from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
from rectified_flow import TimeConditionalRectifiedFlow
from skipconn import TimeConditionalUNet
from utils import seed_everything, device
import mediapy as media


seed_everything(42)

train_dataloader = DataLoader(
    MNIST(
        root="data",
        train=True,
        download=False,  # Set to False since the dataset is already downloaded
        transform=ToTensor(),
    ),
    batch_size=128,
    shuffle=True,
)

test_dataloader = DataLoader(
    MNIST(
        root="data",
        train=False,
        download=False,  # Set to False since the dataset is already downloaded
        transform=ToTensor(),
    ),
    batch_size=128,
    shuffle=False,
)

n_epochs = 20

rf = TimeConditionalRectifiedFlow(unet=TimeConditionalUNet(1, num_hiddens=64).to(device), num_ts=300).to(device)

optimizer = torch.optim.Adam(rf.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1**(1/n_epochs))
# just use a simple scheduler
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8, 12, 16, 20], gamma=0.5)
# multi-step scheduler: 0-19: 1e-3, 20-39: 1e-4, 40-49: 1e-5

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
rf.unet.apply(initialize_weights)

losses = []
epochs_to_sample = [4, 19, 34, 49]
# epochs_to_sample = []

for epoch in range(n_epochs):
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True, desc=f"Epoch {epoch+1}/{n_epochs}")
    rf.train()
    rf.unet.train()
    for i, (x, _) in loop:
        trainstep = i + epoch * len(train_dataloader)
        seed_everything(trainstep)
        optimizer.zero_grad()
        loss = rf.forward(x.to(device))
        loss.backward()
        with torch.no_grad():
            if trainstep % 10 == 0:
                losses.append(loss.item())
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    with torch.no_grad():
        if epoch in epochs_to_sample:
            animation_frames = []
            animation_caches_each_sample = [[] for _ in range(10)]
            rf.eval()
            rf.unet.eval()
            # (width = 10*28, height=4*28)
            canvas = torch.zeros((4*28, 10*28)).to(device) # (4*28, 10*28)
            for i in range(10): 
                for j in range(4):
                    x, anim = rf.sample((28, 28), seed=i*4+j, animation_steps=50)
                    canvas[j*28:(j+1)*28, i*28:(i+1)*28] = x.squeeze() # x.squeeze(): (1, 1, 28, 28) -> (1, 28, 28) -> (28, 28)
                    animation_caches_each_sample[i].append(anim)
            plt.imshow(canvas.cpu().numpy(), cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            os.makedirs("output/timecond", exist_ok=True)
            # plt.savefig(f"output/timecond/epoch_{epoch+1}_sample_results.png", bbox_inches='tight', pad_inches=0, dpi=450)
            media.write_image(f"output/timecond/epoch_{epoch+1}_sample_results.png", canvas.cpu().numpy())
            # plt.show()

            # anims
            n_frames = len(animation_caches_each_sample[0][0])

            for frame in range(n_frames):
                canvas = torch.zeros((4*28, 10*28)).to(device)
                for i in range(10):
                    for j in range(4):
                        canvas[j*28:(j+1)*28, i*28:(i+1)*28] = animation_caches_each_sample[i][j][frame].squeeze()
                animation_frames.append(canvas.cpu().numpy())
            # last frame repeated * 5
            for _ in range(5):
                animation_frames.append(animation_frames[-1])

            animation_output_path = f"output/timecond/epoch_{epoch+1}_animation.gif"
            media.write_video(animation_output_path, animation_frames, fps=10, codec='gif')
    scheduler.step()
    loop.close()
    
# test loss
rf.eval()
rf.unet.eval()
test_losses = []
for x, _ in test_dataloader:
    loss = rf.forward(x.to(device))
    test_losses.append(loss.item())
print("Test loss:", sum(test_losses) / len(test_losses))

# training loss curve in log scale
steps = list(range(0, len(losses)*10, 10))
plt.clf()

plt.plot(steps, losses)
plt.yscale("log")
plt.xlabel("Steps")
plt.ylabel("Loss (Log Scale)")
plt.tight_layout()
plt.savefig("output/timecond/training_loss_curve.png", bbox_inches='tight', pad_inches=0, dpi=450)