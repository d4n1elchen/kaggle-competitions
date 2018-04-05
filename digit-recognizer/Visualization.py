import numpy as np
import matplotlib.pyplot as plt
import torchvision

fig = plt.ion()

def imshow(images, labels):
    img = torchvision.utils.make_grid(images)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    print(' '.join('{}'.format(int(labels[j])) for j in range(len(labels))))
