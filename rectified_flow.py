import torch
import torch.nn as nn
import torch.nn.functional as F
from skipconn import TimeConditionalUNet, ClassConditionalUNet
from utils import seed_everything

class TimeConditionalRectifiedFlow(nn.Module):
    def __init__(
        self,
        unet: TimeConditionalUNet,
        num_ts: int = 300,  # Number of timesteps when sampling
    ):
        super().__init__()
        self.unet = unet  # UNet as the flow field
        # UNet usage: unet(x, t / num_ts): (N, C, H, W), (N,) -> (N, C, H, W)
        self.num_ts = num_ts
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # TODO: Nonlinear interpolation of Xt = alpha * X0 + (1 - alpha) * X1, from paper https://arxiv.org/pdf/2209.03003 chapter 2.3

    def forward(self, x1: torch.Tensor, t_sample='continuous') -> torch.Tensor:
        """
        Args:
            x1: (N, C, H, W) target distribution samples.
            t_sample: 'continuous' or 'discrete' or 'sigmoid' for sampling timesteps.

        Returns:
            torch.Tensor: Scalar transport cost.
        """
        self.unet.train()
        self.train()
        N, C, H, W = x1.shape

        # Sample x0 from the standard normal distribution
        x0 = torch.randn(N, C, H, W, device=self.device)  # (N, C, H, W)

        # Sample timesteps from the uniform distribution [0,1]
        if t_sample == 'continuous':
            t = torch.rand(N, device=self.device) # t ~ U(0, 1)
        elif t_sample == 'sigmoid':
            # t0 ~ N(0, 1), t = sigmoid(t0)
            t0 = torch.randn(N, device=self.device)
            t = torch.sigmoid(t0)
        elif t_sample == 'discrete': # sample t from discrete distribution {0, 1/num_ts, 2/num_ts, ..., 1}
            t = torch.linspace(0, 1, self.num_ts, device=self.device)[torch.randint(0, self.num_ts, (N,))] # t ~ U{0, 1/num_ts, 2/num_ts, ..., 1}
        else:
            raise ValueError(f"Invalid t_sample: {t_sample}")
        
        # xt[i] = t[i] * x1[i] + (1 - t[i]) * x0[i]
        xt = t.view(-1, 1, 1, 1) * x1 + (1 - t).view(-1, 1, 1, 1) * x0
        pred = self.unet(xt, t)  # (N, C, H, W)
        diff = x1 - x0 # FIXED: NOT TO DIVIDE BY T
        # Compute the transport cost as the L2 norm of the difference
        return F.mse_loss(pred, diff, reduction='mean')


    @torch.inference_mode()
    def sample(
        self,
        img_wh: tuple[int, int],
        seed: int = 42,
        animation_steps: int = 50,
    ) -> torch.Tensor:
        """
        Sample an image from the learned flow field using Euler's method.
        
        Args:
            img_wh: Tuple of (H, W) specifying the image resolution.
            seed: Random seed for reproducibility.
            animation_steps: Number of steps for the animation.

        Returns:
            torch.Tensor: Generated image tensor of shape (N, C, H, W).
        """
        self.eval()
        self.unet.eval()
        seed_everything(seed)
        H, W = img_wh
        # Initialize samples from the base distribution (Gaussian noise)
        anim = []
        x = torch.randn(1, 1, H, W, device=self.device)  # (N=1, C=1, H, W)
        anim.append(x.detach().cpu().clone())
        # Integrate flow ODE forward from t=0 to t=1
        t_steps = torch.linspace(0, 1, self.num_ts, device=self.device)  # (num_ts,)
        for i, t in enumerate(t_steps[:-1]):
            # Compute flow field f(x, t)
            dx = self.unet(x, t)
            # Update x using Euler's method
            x = x + dx / self.num_ts
            if i % animation_steps == 0:
                anim.append(x.detach().cpu().clone())
        return x, anim + [x.detach().cpu().clone()]
    
class ClassConditionalRectifiedFlow(nn.Module):
    def __init__(
        self,
        unet: ClassConditionalUNet,
        num_ts: int = 1000,  # Number of timesteps when sampling
        p_uncond: float = 0.1,  # Probability of unconditional sampling
    ):
        super().__init__()
        self.unet = unet  # UNet as the flow field
        # UNet usage: unet(x, t / num_ts): (N, C, H, W), (N,) -> (N, C, H, W)
        self.num_ts = num_ts
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.p_uncond = p_uncond
        
    def forward(self, x1: torch.Tensor, y1: torch.Tensor, t_sample='continuous') -> torch.Tensor:
        """
        Args:
            x1: (N, C, H, W) target distribution samples.
            y1: (N,) target distribution labels.
            t_sample: 'continuous' or 'discrete' or 'sigmoid' for sampling timesteps.

        Returns:
            torch.Tensor: Scalar transport cost.
        """
        self.unet.train()
        self.train()
        N, C, H, W = x1.shape
        x1, y1 = x1.to(self.device), y1.to(self.device)
        # Sample x0 from the standard normal distribution
        x0 = torch.randn(N, C, H, W, device=self.device)
        # Sample timesteps from the uniform distribution [0,1]
        if t_sample == 'continuous':
            t = torch.rand(N, device=self.device)
        elif t_sample == 'sigmoid':
            t0 = torch.randn(N, device=self.device)
            t = torch.sigmoid(t0)
        elif t_sample == 'discrete':
            t = torch.linspace(0, 1, self.num_ts, device=self.device)[torch.randint(0, self.num_ts, (N,))]
        else:
            raise ValueError(f"Invalid t_sample: {t_sample}")
        
        # xt[i] = t[i] * x1[i] + (1 - t[i]) * x0[i]
        xt = t.view(-1, 1, 1, 1) * x1 + (1 - t).view(-1, 1, 1, 1) * x0
        mask = torch.rand(N, device=self.device) > self.p_uncond # with probability p_uncond, mask = 0
        pred = self.unet.forward(xt, y1, t, mask)  # (N, C, H, W)
        diff = x1 - x0
        return F.mse_loss(pred, diff, reduction='mean')
    
    @torch.inference_mode()
    def sample(
        self,
        c: torch.Tensor,
        img_wh: tuple[int, int],
        guidance_scale: float = 5.0,
        seed: int = 42,
        animation_steps: int = 50,
    ) -> torch.Tensor:
        """
        Sample an image from the learned flow field using Euler's method.
        
        Args:
            img_wh: Tuple of (H, W) specifying the image resolution.
            seed: Random seed for reproducibility.
            guidance_scale: Scale factor for the guidance vector in CFG.
            animation_steps: Number of steps for the animation.

        Returns:
            torch.Tensor: Generated image tensor of shape (N, C, H, W).
        """
        self.eval()
        self.unet.eval()
        seed_everything(seed)
        c = c.to(self.device)
        H, W = img_wh
        # Initialize samples from the base distribution (Gaussian noise)
        anim = []
        x = torch.randn(1, 1, H, W, device=self.device)
        anim.append(x.detach().cpu().clone())
        # Integrate flow ODE forward from t=0 to t=1
        t_steps = torch.linspace(0, 1, self.num_ts, device=self.device)
        for i, t in enumerate(t_steps[:-1]):
            # Compute flow field f(x, t)
            dx_c = self.unet(x, c, t)
            unc_mask = torch.zeros_like(c, device=self.device)
            dx_unc = self.unet(x, c, t, unc_mask)
            dx = (1 - guidance_scale) * dx_unc + guidance_scale * dx_c
            # Update x using Euler's method
            x = x + dx / self.num_ts
            if i % animation_steps == 0:
                anim.append(x.detach().cpu().clone())
        return x, anim + [x.detach().cpu().clone()]