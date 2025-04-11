import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import seed_everything

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Kernel size: 3
        # stride: 1
        # padding: 1
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1) # (N, Ci, H, W) -> (N, Co, H, W)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activ = nn.GELU()

        # self.residual = (
        #     nn.Conv2d(in_channels, out_channels, 1) # (N, Ci, H, W) -> (N, Co, H, W)
        #     if in_channels != out_channels
        #     else nn.Identity()
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv2d(3,1,1)+BN+GELU
        return self.activ(self.bn(self.conv(x)))



class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1) # (N, Ci, H, W) -> (N, Co, H//2, W//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activ = nn.GELU()
        # self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2) # identity won't work here, so we need to use a conv layer to learn the identity mapping
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activ(self.bn(self.conv(x)))


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1) # (N, Ci, H, W) -> (N, Co, H*2, W*2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activ = nn.GELU()
        # self.residual = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1) # identity won't work here too

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activ(self.bn(self.tconv(x))) 


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.activ = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activ(self.avgpool(x))



class Unflatten(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=7, stride=7, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.activ = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activ(self.bn(self.tconv(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels) # (N, Ci, H, W) -> (N, Co, H, W)
        self.conv2 = Conv(out_channels, out_channels) # (N, Ci, H, W) -> (N, Co, H, W)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) # identity mapping # (N, Ci, H, W) -> (N, Co, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + self.shortcut(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dconv = DownConv(in_channels, out_channels)
        self.conv = ConvBlock(out_channels, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2) # identity mapping # (N, Ci, H, W) -> (N, Co, H//2, W//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.dconv(x)) + self.shortcut(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.uconv = UpConv(in_channels, out_channels)
        self.conv = ConvBlock(out_channels, out_channels)
        self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1) # identity mapping # (N, Ci, H, W) -> (N, Co, H*2, W*2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.uconv(x)) + self.shortcut(x)

class FCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.activ = nn.GELU()
        self.linear2 = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activ(self.linear1(x)))

class TimeConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1, # dimension of t
        num_hiddens: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.conv0 = Conv(in_channels, num_hiddens)
        self.down1 = DownBlock(num_hiddens, num_hiddens)
        self.down2 = DownBlock(num_hiddens, 2 * num_hiddens)
        self.flatten = Flatten()
        self.unflatten = Unflatten(2 * num_hiddens)
        self.up1 = UpBlock(4 * num_hiddens, num_hiddens)
        self.up2 = UpBlock(2 * num_hiddens, num_hiddens)
        self.conv1 = Conv(2 * num_hiddens, num_hiddens)
        self.conv2 = nn.Conv2d(num_hiddens, in_channels, 3, stride=1, padding=1)
        self.fc1 = FCBlock(1, 2 * num_hiddens) # input: (N,) output: (N, D, 1, 1)
        self.fc2 = FCBlock(1, num_hiddens) # inp: (N,) out: (N, 2D, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.
            t: (N,) normalized time tensor.

        Returns:
            (N, C, H, W) output tensor.
        """
        assert x.shape[-2:] == (28, 28), "Expect input shape to be (28, 28)."
        x = self.conv0(x) # (D, 28, 28)
        xdown1 = self.down1(x) # (D, 14, 14)
        xdown2 = self.down2(xdown1) # (2D, 7, 7)
        flat = self.flatten(xdown2) # (2D, 1, 1)
        unflat = self.unflatten(flat) # (2D, 7, 7)
        t = t.view(-1, 1) # (N, 1)
        fc1t = self.fc1(t) # (N, D)
        fc1t = fc1t.unsqueeze(-1).unsqueeze(-1) # (N, D, 1, 1) # einops.repeat(fc2t, "n d -> n d h w", h=1, w=1)
        fc2t = self.fc2(t) # (N, 2D) # fc2t = einops.repeat(fc2t, "n d -> n d h w", h=1, w=1)
        fc2t = fc2t.unsqueeze(-1).unsqueeze(-1) # (N, 2D, 1, 1)
        unflat = unflat + fc1t # (N, D, 7, 7), broadcast

        xup1 = self.up1(torch.cat([unflat, xdown2], dim=1)) # (4D, 7, 7) -> (D, 14, 14)
        xup1 = xup1 + fc2t # (N, 2D, 7, 7), broadcast
        xup2 = self.up2(torch.cat([xup1, xdown1], dim=1)) # (2D, 14, 14) -> (D, 28, 28)
        x = self.conv1(torch.cat([xup2, x], dim=1)) # (2D, 28, 28) -> (D, 28, 28)
        x = self.conv2(x) # (D, 28, 28)
        return x

class ClassConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_hiddens: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.conv0 = Conv(in_channels, num_hiddens)
        self.down1 = DownBlock(num_hiddens, num_hiddens)
        self.down2 = DownBlock(num_hiddens, 2 * num_hiddens)
        self.flatten = Flatten()
        self.unflatten = Unflatten(2 * num_hiddens)
        self.up1 = UpBlock(4 * num_hiddens, num_hiddens)
        self.up2 = UpBlock(2 * num_hiddens, num_hiddens)
        self.conv1 = Conv(2 * num_hiddens, num_hiddens)
        self.conv2 = nn.Conv2d(num_hiddens, in_channels, 3, stride=1, padding=1)
        self.fc1_t = FCBlock(1, 2 * num_hiddens) # input: (N,) output: (N, 2D, 1, 1)
        self.fc2_t = FCBlock(1, num_hiddens) # inp: (N,) out: (N, D, 1, 1)
        self.fc1_c = FCBlock(num_classes, 2 * num_hiddens) # input: (N, N_CLASSES) output (N, 2D, 1, 1)
        self.fc2_c = FCBlock(num_classes, num_hiddens) # input: (N, N_CLASSES) output (N, D, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.
            c: (N,) int64 condition tensor.
            t: (N,) normalized time tensor.
            mask: (N,) mask tensor. If not None, mask out condition when mask == 0.

        Returns:
            (N, C, H, W) output tensor.
        """
        assert x.shape[-2:] == (28, 28), "Expect input shape to be (28, 28)."
        x = self.conv0(x) # (D, 28, 28)
        xdown1 = self.down1(x) # (D, 14, 14)
        xdown2 = self.down2(xdown1) # (2D, 7, 7)
        flat = self.flatten(xdown2) # (2D, 1, 1)
        unflat = self.unflatten(flat) # (2D, 7, 7)
        t = t.view(-1, 1) # (N, 1)
        c_onehot = F.one_hot(c, num_classes=self.num_classes).to(device).float() # (N, N_CLASSES_)
        # for mask[i] = 0, c_onehot[i] = 0 too
        if mask is not None:
            c_onehot = c_onehot * mask.view(-1, 1)
        t1 = self.fc1_t(t) # (N, D)
        t1 = t1.unsqueeze(-1).unsqueeze(-1) # (N, D, 1, 1) # einops.repeat(fc2t, "n d -> n d h w", h=1, w=1)
        t2 = self.fc2_t(t) # (N, 2D) # fc2t = einops.repeat(fc2t, "n d -> n d h w", h=1, w=1)
        t2 = t2.unsqueeze(-1).unsqueeze(-1) # (N, 2D, 1, 1)
        c1 = self.fc1_c(c_onehot) # (N, D)
        c1 = c1.unsqueeze(-1).unsqueeze(-1) # (N, D, 1, 1)

        # if not none, c1 in that place = all 1

        unflat = c1 * unflat + t1 # (N, D, 7, 7), broadcast
        c2 = self.fc2_c(c_onehot) # (N, 2D) # fc2t = einops.repeat(fc2t, "n d -> n d h w", h=1, w=1)
        c2 = c2.unsqueeze(-1).unsqueeze(-1) # (N, 2D, 1, 1)
        xup1 = self.up1(torch.cat([unflat, xdown2], dim=1)) # (4D, 7, 7) -> (D, 14, 14)
        xup1 = c2 * xup1 + t2 # (N, 2D, 7, 7), broadcast
        xup2 = self.up2(torch.cat([xup1, xdown1], dim=1)) # (2D, 14, 14) -> (D, 28, 28)
        x = self.conv1(torch.cat([xup2, x], dim=1)) # (2D, 28, 28) -> (D, 28, 28)
        x = self.conv2(x) # (D, 28, 28)
        return x