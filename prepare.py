import os
import torchvision
from torchvision import datasets, transforms

# download mnist
# Create the data/ directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download MNIST dataset into the data/ directory
print("Downloading MNIST dataset...")
transform = transforms.Compose([transforms.ToTensor()])
datasets.MNIST(root="data", train=True, download=True, transform=transform)
datasets.MNIST(root="data", train=False, download=True, transform=transform)

# Create the output/ directory if it doesn't exist
os.makedirs("output", exist_ok=True)