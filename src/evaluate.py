import torch

def evaluate_model(model, data_loader, device):
 model.eval()
 correct = 0
 total = 0
 with torch.no_grad():
     for imgs, annotations in data_loader:
         imgs = [img.to(device) for img in imgs]
         predictions = model(imgs)

         
         total += len(annotations)
         correct += sum(1 for pred, ann in zip(predictions, annotations) if len(pred['boxes']) == len(ann['boxes']))
 
 return correct / total
