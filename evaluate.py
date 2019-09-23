from api import IntegratedApi

import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

if __name__ == '__main__':
    thresholds = []
    for i in range(1, 10):
        thresholds.append(i/10)

    for threshold in thresholds:
        # TODO apply api when size mismatch problem is solved
        # api = IntegratedApi(threshold=threshold)
        # result = api.detect(some_image)
        pass

