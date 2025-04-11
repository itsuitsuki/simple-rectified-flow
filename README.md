# simple-rectified-flow
A simple implementation for Rectified Flow. Some of the framework references COMPSCI 180 FA24, UC Berkeley.

# Requirements
Please refer to & install by
```sh
conda create -n rf python=3.12
conda activate rf
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 # tailor the CUDA version to your needs
pip install -r requirements.txt
```

# Usage
## Before training
```sh
python prepare.py
```
## Training of Rectified Flow by MNIST
### Time-conditional Rectified Flow (Unconditional)
```sh
python train_rf_timecond.py
```

### Class-conditional Rectified Flow
```sh
python train_rf_classcond.py
```

