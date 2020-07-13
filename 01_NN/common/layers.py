import numpy as np


class Affine:
  def __init__(self, W, b):
    self.params, self.grads = [W, b], []
    self.out = None

  def forward(self, x):
    W, b = self.params
    self.out = np.matmul(x, W) + b
    return self.out

  def backward(self, dout):
    None


class Sigmoid:
  def __init__(self):
    self.params, self.grads = [], []
    self.out = None

  def forward(self, x):
    out = 1 / (1 + np.exp(-1))
    self.out = out
    return out
  
  def backward(self, dout):
    dx = dout * (1.0 - self.out) * self.out
    return dx




