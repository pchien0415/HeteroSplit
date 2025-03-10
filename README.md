# D2D-Assisted Split Learning Approach for Model-Heterogeneous Federated Learning

## Environment Setup

### System Requirements
* Python 3.7
* PyTorch 1.11.0

### Installing Dependencies
Use the provided environment file:
```
conda env create -f environment.yml
```

**Note**: Ensure you install the CUDA version corresponding to your hardware.

## Training
Run the training:
```
python main.py --algo propose --data cifar10
```
or using make to run
```
make all
```

## Parameters
All hyper-parameters are controlled from ``args.py``.




