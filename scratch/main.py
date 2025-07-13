import matplotlib.pyplot as plt
from dataset.mnist import load_mnist


def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    img = x_train[0].reshape(28, 28)
    label = t_train[0]

    plt.imshow(img, cmap='gray')
    plt.title(label)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
