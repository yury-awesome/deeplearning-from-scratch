import matplotlib.pyplot as plt
from scratch.dataset.mnist import load_mnist


def img_show(img, label):
    """
    Function to display the image from the MNIST dataset.
    """
    img = img.reshape(28, 28)
    
    plt.imshow(img, cmap='gray')
    plt.title(label)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    img, label = x_train[0], t_train[0]
    img_show(img, label)
