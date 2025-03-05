# D2D-Assisted Split Learning Approach for Model-Heterogeneous Federated Learning
Code for the following paper:

## Environment Setup

### System Requirements
* Python 3.7
* PyTorch 1.11.0

### Installing Dependencies
use the provided environment file:
```
conda env create -f environment.yml
```

**Note**: Ensure you install the CUDA version corresponding to your hardware (e.g., CUDA 11.6).


## Usage

## Training
Run the training:
```
python main.py --algo propose
```
or using make to run
```
make all
```

### Parameters
All hyper-parameters are controlled from ``args.py``.




