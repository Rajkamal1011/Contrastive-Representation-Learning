import torch
from argparse import Namespace
from typing import Union, Tuple, List

import ContrastiveRepresentation.pytorch_utils as ptu
from utils import get_data_batch, get_contrastive_data_batch
from LogisticRegression.model import LinearModel
from LogisticRegression.train_utils import fit_model as fit_linear_model,\
    calculate_loss as calculate_linear_loss,\
    calculate_accuracy as calculate_linear_accuracy

import torch.nn.functional as F  # Import the functional API

def calculate_loss(y_logits: torch.Tensor, y: torch.Tensor) -> float:
    '''
    Calculate the negative log likelihood loss given the softmax logits and the labels.

    Args:
        y_logits: torch.Tensor, the raw scores output from the model (before softmax).
        y: torch.Tensor, the ground truth labels.

    Returns:
        loss: float, the negative log likelihood loss.
    '''

    # Apply log softmax to convert softmax logits to log probabilities
    log_probs = F.log_softmax(y_logits, dim=1)  # dim=1 as we're dealing with batches

    # Calculate the negative log likelihood loss
    loss = F.nll_loss(log_probs, y)  # NLLLoss expects log probabilities as input

    # Return the loss value
    return loss.item()  # Use .item() to extract the scalar loss value as a Python float

def calculate_accuracy(y_logits: torch.Tensor, y: torch.Tensor) -> float:
    '''
    Calculate the accuracy of the model on the given data.

    Args:
        y_logits: torch.Tensor, softmax logits from the model.
        y: torch.Tensor, labels (indices for classification tasks).

    Returns:
        acc: float, accuracy of the model as a percentage.
    '''
    # Convert softmax logits to predicted class indices
    _, predictions = torch.max(y_logits, dim=1)
    
    # Compare predictions with the true labels and count the correct ones
    correct_count = (predictions == y).sum().item()
    
    # Calculate the accuracy
    acc = correct_count / y.size(0) * 100  # Convert to percentage
    
    return acc

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.modules.loss import TripletMarginLoss
import matplotlib.pyplot as plt

def fit_contrastive_model(
        encoder: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        num_iters: int = 1000,
        batch_size: int = 256,
        learning_rate: float = 1e-3
) -> list:
    '''
    Fit the contrastive model.

    Args:
    - encoder: torch.nn.Module, the encoder model
    - X: torch.Tensor, features
    - y: torch.Tensor, labels
    - num_iters: int, number of iterations for training
    - batch_size: int, batch size for training
    - learning_rate: float, learning rate for the optimizer

    Returns:
    - losses: List[float], list of losses at each iteration
    '''
    encoder.train()  # Set the encoder to training mode
    optimizer = Adam(encoder.parameters(), lr=learning_rate)  # Define the optimizer
    loss_fn = TripletMarginLoss(margin=1.0)  # Define the loss function
    losses = []
    triple = get_contrastive_data_batch(X, y, batch_size)

    for i in range(num_iters):
        # Get a batch of anchor, positive, and negative samples
        X_a, X_p, X_n = next(triple)
        # X_a, X_p, X_n = get_contrastive_data_batch(X, y, batch_size)

        # Convert numpy arrays to PyTorch tensors if they are not already
        if not isinstance(X_a, torch.Tensor):
            X_a = torch.tensor(X_a, dtype=torch.float32)
        if not isinstance(X_p, torch.Tensor):
            X_p = torch.tensor(X_p, dtype=torch.float32)
        if not isinstance(X_n, torch.Tensor):
            X_n = torch.tensor(X_n, dtype=torch.float32)

        device = next(encoder.parameters()).device
        X_a, X_p, X_n = X_a.to(device), X_p.to(device), X_n.to(device)
    
        # Ensure the tensors are on the correct device (CPU or CUDA)
        # X_a, X_p, X_n = X_a.to(encoder.device), X_p.to(encoder.device), X_n.to(encoder.device)

        # Zero the gradients
        optimizer.zero_grad()

        # Compute the embeddings for anchors, positives, and negatives
        v_a = encoder(X_a)
        v_p = encoder(X_p)
        v_n = encoder(X_n)

        # Compute the loss
        loss = loss_fn(v_a, v_p, v_n)
        losses.append(loss.item())

        # Backpropagate and update the encoder weights
        loss.backward()
        optimizer.step()

        # Optional: Print the loss every few iterations
        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")


    return losses


import torch
from typing import Tuple, Union

def evaluate_model(
        encoder: torch.nn.Module,
        classifier: Union[torch.nn.Linear, torch.nn.Module],
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 256,
        is_linear: bool = False
) -> Tuple[float, float]:
    '''
    Evaluate the model on the given data.

    Args:
    - encoder: torch.nn.Module, the encoder model
    - classifier: Union[torch.nn.Linear, torch.nn.Module], the classifier model
    - X: torch.Tensor, images
    - y: torch.Tensor, labels
    - batch_size: int, batch size for evaluation
    - is_linear: bool, whether the classifier is linear

    Returns:
    - loss: float, loss of the model
    - acc: float, accuracy of the model
    '''
    encoder.eval()  # Set the encoder to evaluation mode
    classifier.eval()  # Set the classifier to evaluation mode
    
    total_loss = 0
    total_acc = 0
    total_samples = 0

    with torch.no_grad():  # No need to track gradients for evaluation
        for i in range(0, X.size(0), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            # Get the embeddings from the encoder
            embeddings = encoder(X_batch)
            
            # Pass the embeddings to the classifier
            y_preds = classifier(embeddings)
            
            # Depending on the classifier type, use appropriate functions to calculate loss and accuracy
            if is_linear:
                batch_loss = calculate_linear_loss(y_preds, y_batch)
                batch_acc = calculate_linear_accuracy(y_preds, y_batch)
            else:
                batch_loss = calculate_loss(y_preds, y_batch)
                batch_acc = calculate_accuracy(y_preds, y_batch)
            
            total_loss += batch_loss * X_batch.size(0)
            total_acc += batch_acc * X_batch.size(0)
            total_samples += X_batch.size(0)
    
    # Calculate the average loss and accuracy over all batches
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples

    return avg_loss, avg_acc



import torch
import numpy as np
from typing import Union, List, Tuple
from argparse import Namespace
from LogisticRegression.model import SoftmaxRegression as LinearClassifier

# def calculate_accuracy(y_pred, y_true):
#     """Calculates accuracy given predictions and true labels."""
#     # Assuming y_pred is a tensor of logits and y_true is a tensor of integer labels
#     preds = torch.argmax(y_pred, dim=1)
#     correct = torch.sum(preds == y_true).item()
#     total = y_true.size(0)
#     return correct / total

import torch.nn as nn
# Define the custom loss function
def calculate_loss_nn(y_logits, y_true):
    """
    Calculate the loss using PyTorch's CrossEntropyLoss for a classification task.

    Parameters:
    - y_logits: The predicted logits from the classifier. Logits are the raw scores output by the last layer of the network.
    - y_true: The ground truth labels.

    Returns:
    - loss: A PyTorch tensor representing the computed loss.
    """
    # print(y_logits.dtype, y_true.dtype)
    criterion = nn.CrossEntropyLoss()  # Initialize the loss function
    loss = criterion(y_logits, y_true)  # Calculate the loss
    return loss


def fit_model(
        encoder: torch.nn.Module,
        classifier: Union[LinearClassifier, torch.nn.Module],
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        args: Namespace
) -> Tuple[List[float], List[float], List[float], List[float]]:
    train_losses, train_accs, val_losses, val_accs = [], [], [], []



# Assuming args.mode, encoder, X_train, X_val, y_train, y_val, and args.batch_size are defined elsewhere

    if args.mode == 'fine_tune_linear':
        print("YES LINEAR fine tune")
        encoder.eval()  # Set encoder to evaluation mode

        # Lists to store embeddings
        train_embeddings = []
        val_embeddings = []

        # Loop over training data in batches and store embeddings
        for i in range(0, X_train.shape[0], args.batch_size):
            X_batch = X_train[i:i+args.batch_size]
            with torch.no_grad():
                embedding = encoder(X_batch).detach().cpu().numpy()
            print(f"batch: {i}")
            train_embeddings.append(embedding)

        # Convert list of arrays to a single 2D array for training embeddings
        X_train_final = np.vstack(train_embeddings)

    # Loop over validation data in batches and store embeddings
        for i in range(0, X_val.shape[0], args.batch_size):
            X_batch = X_val[i:i+args.batch_size]  # Make sure to use X_val here, not X_train
            with torch.no_grad():
                embedding = encoder(X_batch).detach().cpu().numpy()
            print(f"batch: {i}")
            val_embeddings.append(embedding)

    # Convert list of arrays to a single 2D array for validation embeddings
        X_val_final = np.vstack(val_embeddings)

    # Convert labels to NumPy arrays
        y_train_np = y_train.cpu().numpy()
        y_val_np = y_val.cpu().numpy()

        # print("going to fit_linear_model")


    # Assuming fit_linear_model is defined elsewhere and can handle NumPy arrays
        train_losses, train_accs, val_losses, val_accs = fit_linear_model(
        classifier, X_train_final, y_train_np, X_val_final, y_val_np,args.num_iters, args.lr, args.batch_size, args.l2_lambda, args.grad_norm_clip)

        # print("Back from fit_linear_model")
        return train_losses, train_accs, val_losses, val_accs

    else:
        encoder.eval()

        optimizer = torch.optim.Adam(classifier.parameters(), lr = args.lr)

        train_losses, train_accs, val_losses, val_accs = [], [], [], []

        for i in range(args.num_iters + 1):

            classifier.train()

            X_batch, y_batch = get_data_batch(X_train, y_train, args.batch_size)

            optimizer.zero_grad()

            with torch.no_grad():
                z_batch = encoder(X_batch)
            
            # print("z batch shape: ",z_batch.shape)
            y_logits = classifier(z_batch)

            loss = calculate_loss_nn(y_logits, y_batch)

            loss.backward()

            optimizer.step()

            if i%10 == 0:

                train_losses.append(loss.item())

                acc = calculate_accuracy(y_logits, y_batch)

                train_accs.append(acc)

                val_loss, val_acc = evaluate_model(encoder,classifier, X_val, y_val, args.batch_size, is_linear = False)

                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                print(f'Iter {i}/{args.num_iters} - Train loss: {loss.item():.4f} - Train Acc: {acc:.4f} - Val loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        return train_losses, train_accs, val_losses, val_accs



        
