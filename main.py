from api import IntegratedApi
import cv2
import os
import os.path as osp
from glob import glob
import numpy as np
from PIL import Image
from settings import Settings

if __name__ == '__main__':
    settings = Settings()

    if not osp.exists(settings.input_dir):
        raise FileNotFoundError(f"{settings.input_dir} directory not exists.")
    os.makedirs(settings.output_dir, exist_ok=True)

    print('Creating networks and loading parameters')
    api = IntegratedApi(settings)
    print('Preparing detector...')
    result = api.detect(np.array(Image.open(osp.join(settings.input_dir, 'car.jpg'))))

    print(result)
