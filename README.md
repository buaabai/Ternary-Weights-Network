# Ternary-Weights-Network
This is a Pytorch implementation of [Ternary-Weights-Network](https://arxiv.org/abs/1605.04711) for the MNIST dataset.The model structure is LeNet-5. The dataset is provided by [torchvision](https://pytorch.org/docs/master/torchvision/). And here are two ways I used to ternarize the weights, which correspond to `main.py` and `second_main.py`.

# Requirements
- Python, Numpy
- Pytorch 0.3.1

# Usage

    $ git clone https://github.com/buaabai/Ternary-Weights-Network
    $ python main.py --epochs 100
    $ python second_main.py --epochs 100

You can use

    $ python main.py -h

to check other parmeters.

# How to get Ternary Weights

Here are two ways to get ternary weights. In the `main.py`, I use TernaryLinear layer and TernaryConv2d layer in the `model.py`. Both these two layers ternarize their weights during forward computing. In the `second_main.py`, I first use a normal LeNet-5 model, but the weights in the model were ternarized before the forward computing, and after the forward computing, the full precision weights are restored for the update operation. However, the run time of these two ways are both long(`second_main.py` is faster than `main.py`). And both of them run on a M40 gpu of NVIDIA. I don't know the reason.





