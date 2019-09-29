from api import IntegratedApi
import cv2
import os
from settings import Settings

if __name__ == '__main__':
    # load target image
    root = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(root, "test_photos")

    settings = Settings()
    
    api = IntegratedApi(settings)
    images = [f for f in os.listdir(root) if f.endswith(".jpg")]
    for image_path in images:
        image_path = os.path.join(root, image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        cv2.imshow('inference_target', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Inference
        masked_image = api.mask(image, settings)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('inferenced', masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
