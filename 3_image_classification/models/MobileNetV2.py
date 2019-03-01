#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 22:18:12 2019

@author: changlinjiang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    
    def __init__(self, in_channel, out_plane, expansion, )