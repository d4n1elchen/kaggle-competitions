import numpy as np
import matplotlib.pyplot as plt
import torchvision

def imshow(images, labels=None):
    img = torchvision.utils.make_grid(images)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if labels is not None:
        print(' '.join('{}'.format(int(labels[j])) for j in range(len(labels))))
    plt.show()
