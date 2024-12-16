from dataset import MaskDataset
from model import get_model_instance_segmentation
from train import train_model
from visualize import plot_image

def main():

 img_dir = "/path/to/images"
 annotations_dir = "/path/to/annotations"
 
 transform = transforms.Compose([transforms.ToTensor()])
 dataset = MaskDataset(img_dir, annotations_dir, transform)

 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 model = get_model_instance_segmentation(num_classes=3)
 
 trained_model = train_model(dataset, model, device)
 torch.save(trained_model.state_dict(), "model.pth")

if __name__ == "__main__":
 main()
