from collections import Counter
from typing import TypeVar, DefaultDict, List, Dict, Mapping

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

# TypeVar definitions for commonly used PyTorch classes

# ModelType is used for any class derived from torch.nn.Module
ModelType = TypeVar("ModelType", bound=nn.Module)

# OptimType is used for any class derived from torch.optim.Optimizer
OptimType = TypeVar("OptimType", bound=optim.Optimizer)

# SchedulerType is used for any class derived from torch.optim.lr_scheduler._LRScheduler
SchedulerType = TypeVar("SchedulerType", bound=_LRScheduler)

# DeviceType is used for instances of torch.device
DeviceType = TypeVar("DeviceType", bound=torch.device)

# DataIterType is used for instances of torch.utils.data.DataLoader
DataIterType = TypeVar("DataIterType", bound=DataLoader)

Captions = DefaultDict[str, List[List[str]]]
ImagesAndCaptions = Dict[str, Captions]