import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from torchvision import transforms

class IMAGE_Dataset(Dataset):
    """
    Custom Dataset of IMAGE_Dataset, which is used for train and test
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.abspath(root_dir)
        self.x = []
        self.y = []
        self.filename = []
        self.transform = transform
        self.num_classes = 0
        self.className = os.listdir(self.root_dir)      
        
        self.className.sort()
        self.className.sort(key=lambda x:x)
        
        for i, _dir in enumerate(self.className):
            file_list = os.listdir(os.path.join(self.root_dir, _dir))
            for j, file in enumerate(file_list):
                try:
                    np.asarray(Image.open(os.path.join(self.root_dir, _dir, file)))
                    self.x.append(os.path.join(self.root_dir, _dir, file))
                    self.y.append(i)
                    self.filename.append(file_list[j])
                except:
                    continue
            self.num_classes += 1
    
    def __len__(self):
        return len(self.x)


    def contrast_enhance(self, img):
        contrast = 10

        pixel_val = np.asarray(img, dtype='int32')
        if np.average(pixel_val) < 30:
            image_contrasted = ImageEnhance.Contrast(img).enhance(contrast)
        else:
            image_contrasted = img
        return image_contrasted


    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')
        # image = self.contrast_enhance(image)

        if self.transform:
            image = self.transform(image)
        return image, self.y[index]



class Inference_Dataset(Dataset):
    """
    Custom Dataset which is used for inference.
    In this dataset, the label is all 0, cause inference data has no label.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.x = []
        self.filename = []
        self.transform =  transforms.Compose([
                        transforms.ToTensor(),])

        for name in os.listdir(root_dir):
            try:
                Image.open(os.path.join(root_dir, name)).convert('RGB')
                self.x.append(os.path.join(root_dir, name))
                self.filename.append(name)
            except:
                continue

    
    def __len__(self):
        return len(self.x)


    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Inference has no label
        label = 0

        return image, label