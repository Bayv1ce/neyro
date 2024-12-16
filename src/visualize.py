import matplotlib.pyplot as plt

def plot_image(img_tensor, annotation):
 fig, ax = plt.subplots(1)
 img = img_tensor.cpu().data.permute(1, 2, 0)
 ax.imshow(img)
 
 for box in annotation['boxes']:
     xmin, ymin, xmax, ymax = box
     rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
     ax.add_patch(rect)
 plt.show()
