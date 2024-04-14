import torch
import numpy as np
from typing import Tuple
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def get_data(
        data_path: str = 'data/cifar10_train.npz', is_linear: bool = False,
        is_binary: bool = False, grayscale: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load CIFAR-10 dataset from the given path and return the images and labels.
    If is_linear is True, the images are reshaped to 1D array.
    If grayscale is True, the images are converted to grayscale.

    Args:
    - data_path: string, path to the dataset
    - is_linear: bool, whether to reshape the images to 1D array
    - is_binary: bool, whether to convert the labels to binary
    - grayscale: bool, whether to convert the images to grayscale

    Returns:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    '''
    data = np.load(data_path)
    X = data['images']
    try:
        y = data['labels']
    except KeyError:
        y = None
    X = X.transpose(0, 3, 1, 2)
    if is_binary:
        idxs0 = np.where(y == 0)[0]
        idxs1 = np.where(y == 1)[0]
        idxs = np.concatenate([idxs0, idxs1])
        X = X[idxs]
        y = y[idxs]
    if grayscale:
        X = convert_to_grayscale(X)
    if is_linear:
        X = X.reshape(X.shape[0], -1)
    
    # HINT: rescale the images for better (and more stable) learning and performance
    mean = np.mean(X)
    std_dev = np.std(X)
    X = (X-mean)/(std_dev + 1e-15)
    # mx = np.max(X)
    # mn = np.min(X)
    # X = (mx - X)/(mx - mn)
    # X = X/255.0

    return X, y


def convert_to_grayscale(X: np.ndarray) -> np.ndarray:
    '''
    Convert the given images to grayscale.

    Args:
    - X: np.ndarray, images in RGB format

    Returns:
    - X: np.ndarray, grayscale images
    '''
    return np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])


def train_test_split(
        X: np.ndarray, y: np.ndarray, test_ratio: int = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Split the given dataset into training and test sets.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - test_ratio: float, ratio of the test set

    Returns:
    - X_train: np.ndarray, training images
    - y_train: np.ndarray, training labels
    - X_test: np.ndarray, test images
    - y_test: np.ndarray, test labels
    '''
    assert test_ratio < 1 and test_ratio > 0

    #Adding new code
    #Shuffle the dataset
    # print(X)
    # print(y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    #Split the dataset
    test_size = int(X.shape[0] * test_ratio)
    X_train = X[:-test_size]
    y_train = y[:-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]

    #End of addition of new code
    #raise NotImplementedError('Split the dataset here')
    
    return X_train, y_train, X_test, y_test


def get_data_batch(
        X: np.ndarray, y: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Get a batch of the given dataset.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - batch_size: int, size of the batch

    Returns:
    - X_batch: np.ndarray, batch of images
    - y_batch: np.ndarray, batch of labels
    '''
    #Adding new code : 
    #Ensure batch size does not exist the dataset size
    # print(X.shape)
    batch_size = min(batch_size, X.shape[0])

    #Ger random indices of the batch size without replacement from the dataset
    idxs = np.random.choice(X.shape[0], size=batch_size, replace=False)

    #End of addtion of new code
    # idxs = # TODO: get random indices of the batch size without replacement from the dataset
    return X[idxs], y[idxs]


import numpy as np

import numpy as np
import torch

def get_contrastive_data_batch(X, y, batch_size):
    '''
    Get a batch of the given dataset for contrastive learning.

    Args:
    - X: np.ndarray or torch.Tensor, images
    - y: np.ndarray or torch.Tensor, labels
    - batch_size: int, size of the batch

    Yields:
    - X_a: np.ndarray, batch of anchor samples
    - X_p: np.ndarray, batch of positive samples
    - X_n: np.ndarray, batch of negative samples
    '''

    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    while True:  # This makes the generator infinite
        # Initialize batches
        X_a = np.zeros((batch_size,) + X.shape[1:])
        X_p = np.zeros((batch_size,) + X.shape[1:])
        X_n = np.zeros((batch_size,) + X.shape[1:])
        
        for i in range(batch_size):
            # Randomly select an anchor
            anchor_idx = np.random.randint(0, X.shape[0])
            anchor_label = y[anchor_idx]

            # Find a positive sample (same label but different image)
            positive_idxs = np.where(y == anchor_label)[0]
            positive_idxs = positive_idxs[positive_idxs != anchor_idx]  # Exclude the anchor
            positive_idx = np.random.choice(positive_idxs)

            # Find a negative sample (different label)
            negative_idxs = np.where(y != anchor_label)[0]
            negative_idx = np.random.choice(negative_idxs)

            # Assign samples to batches
            X_a[i] = X[anchor_idx]
            X_p[i] = X[positive_idx]
            X_n[i] = X[negative_idx]

        yield X_a, X_p, X_n




def plot_losses(
        train_losses: list, val_losses: list, title: str
) -> None:
    '''
    Plot the training and validation losses.

    Args:
    - train_losses: list, training losses
    - val_losses: list, validation losses
    - title: str, title of the plot
    '''
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig('images/loss.png')
    plt.close()


def plot_accuracies(
        train_accs: list, val_accs: list, title: str
) -> None:
    '''
    Plot the training and validation accuracies.

    Args:
    - train_accs: list, training accuracies
    - val_accs: list, validation accuracies
    - title: str, title of the plot
    '''
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.savefig('images/acc_.png')
    plt.close()

import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(z: torch.Tensor, y: torch.Tensor) -> None:
    '''
    Plot the 2D t-SNE of the given representation.

    Args:
    - z: torch.Tensor, representation
    - y: torch.Tensor, labels
    '''
    # Ensure the 'images' directory exists or change the path according to your directory structure

    # Convert torch.Tensor to numpy array for both z and y
    z_numpy = z.cpu().detach().numpy()  # Convert tensor to numpy array
    y_numpy = y.cpu().detach().numpy() if isinstance(y, torch.Tensor) else y  # Ensure y is a numpy array

    # Perform t-SNE embedding
    tsne = TSNE(n_components=2, random_state=0)
    z_tsne = tsne.fit_transform(z_numpy)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y_numpy, cmap='tab10')
    plt.colorbar(label='Classes')
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.savefig('images/tsne.png')  # Ensure this path exists or is adjusted to your setup
    plt.close()
