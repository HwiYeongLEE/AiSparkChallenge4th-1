# Neural Transformation Learning for Anomaly Detection (NeuTraLAD) - a self-supervised method for anomaly detection
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from .utils import *
import torch
import numpy as np
import pandas as pd

def load_data(data, cls):

    train = np.array(data[0][data[0].type==cls].iloc[:, :-1])
    test = np.array(data[1][data[1].type==cls].iloc[:, :-1])
    trainset = CustomDataset(train)
    testset = CustomDataset(test)
    return [trainset, testset]
