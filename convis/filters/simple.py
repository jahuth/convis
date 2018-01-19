import numpy as np
import uuid
from torch import nn
import torch

from ..base import Layer
from ..filters import TimePadding, Conv1d, Conv2d, Conv3d, Delay, VariableDelay, TIME_DIMENSION, L, LN, TemporalLowPassFilterRecursive, TemporalHighPassFilterRecursive, SpatialRecursiveFilter, SmoothConv
