import sys

import numpy as np


class WeightSum:
  def __init__(self):
    None
  
  def forward(self, hs, a):
    None
  
  def backward(self, dc):
    None

class AttentionWeight:
  def __init__(self):
    self.params, self.grads = [], []
    self.softmax = Softmax()
    self.cache = None

  def forward(self, hs, h):
    N, T, H = hs.shape
  
  def backward(self, da):
    hs, hr = self.cache


class Attention:
  def __init__(self):
    None

  def forward(self, hs, h):
    None
  
  def backward(self, dout):
    None


class TimeAttention:
  def __init__(self):
    None

  def forward(self, hs, h):
    None
  
  def backward(self, dout):
    None