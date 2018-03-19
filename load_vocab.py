from __future__ import division

import argparse
import glob
import os
import sys
import random

import torch
import torch.nn as nn
from torch import cuda

voc = torch.load("./glove_experiments/data.vocab.pt")
