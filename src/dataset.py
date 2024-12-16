import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class MaskDataset(Dataset):
 def __init__(self, img_dir, annotations_dir, transforms=None):
     self.imgs = sorted(os.listdir(img_dir))
     self.annotations = sorted(os.listdir(annotations_dir))
     self.transforms = transforms
     self.img_dir = img_dir
     self.annotations_dir = annotations_dir

 def __getitem__(self, idx):
     img_path = os.path.join(self.img_dir, self.imgs[idx])
     annotation_path = os.path.join(self.annotations_dir, self.annotations[idx])

    
     img = Image.open(img_path).convert("RGB")

     annotation = {"boxes": [], "labels": []}

     if self.transforms:
         img = self.transforms(img)
     
     return img, annotation

 def __len__(self):
     return len(self.imgs)
