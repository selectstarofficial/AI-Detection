from api import IntegratedApi
import cv2
import os


if __name__ == '__main__':
    # load target image
    root = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(root, "test_photos")
    
    images = [f for f in os.listdir(root) if f.endswith(".jpg")]
    for image_path in images:
        image_path = os.path.join(root, image_path)
        image = cv2.imread(image_path)
        cv2.imshow('inference_target', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Inference
        api = IntegratedApi()
        masked_image = api.mask(image, show_boundary=False)

        cv2.imshow('inferenced', masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
