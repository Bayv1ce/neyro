import torch
from torch.utils.data import DataLoader
from model import get_model_instance_segmentation
from dataset import MaskDataset
import torchvision.transforms as transforms

def train_model(dataset, model, device, num_epochs=10, lr=0.005):
 data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

 model.to(device)
 optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9)
 
 for epoch in range(num_epochs):
     model.train()
     for imgs, annotations in data_loader:
         imgs = [img.to(device) for img in imgs]
         annotations = [{k: v.to(device) for k, v in ann.items()} for ann in annotations]

         optimizer.zero_grad()
         loss_dict = model(imgs, annotations)
         losses = sum(loss for loss in loss_dict.values())
         losses.backward()
         optimizer.step()
     print(f"Epoch {epoch+1}, Loss: {losses.item()}")

 return model

if __name__ == "__main__":
 transform = transforms.Compose([transforms.ToTensor()])
 dataset = MaskDataset("/path/to/images", "/path/to/annotations", transform)
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 model = get_model_instance_segmentation(num_classes=3)
 trained_model = train_model(dataset, model, device)
 torch.save(trained_model.state_dict(), "model.pth")
