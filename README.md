# Face/License Plate Recognition
> Inference API

## Usage
1. Import api

```python
from api import IntegratedApi

api = IntegratedApi()
```

2. Generate Masked Image

```python
import cv2

image_path = "..."
base_image = cv2.imread(image_path)
masked_image = api.mask(base_image)
```
