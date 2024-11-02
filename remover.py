import numpy as np
import cv2
import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision import transforms

class PersonRemover():
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.deeplab_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Load the Deeplab model once and keep it ready for reuse
        self.deeplab = self.make_deeplab()

    def make_deeplab(self):
        deeplab = deeplabv3_mobilenet_v3_large(pretrained=True).to(self.device)
        deeplab.eval()
        return deeplab

    def apply_deeplab(self, img):
        # Preprocess and run through Deeplab model
        input_tensor = self.deeplab_preprocess(img)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.deeplab(input_batch)['out'][0]
        output_predictions = output.argmax(0).cpu().numpy()
        return (output_predictions == 15)  # Person class is 15 in COCO dataset

    def remove_person(self, frame):
        # Resize frame for processing if larger than 1024 pixels on any dimension
        k = min(1.0, 1024 / max(frame.shape[0], frame.shape[1]))
        resized_frame = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_LANCZOS4)

        # Apply Deeplab to generate person mask
        mask = self.apply_deeplab(resized_frame)
        
        # Create 3-channel mask to zero-out pixels corresponding to people
        mask_3channel = np.stack([mask] * 3, axis=-1)
        resized_frame[mask_3channel] = 0
        
        # Resize the frame back to original size
        if k < 1.0:
            processed_frame = cv2.resize(resized_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        else:
            processed_frame = resized_frame

        return processed_frame
