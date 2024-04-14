import numpy as np
from typing import Tuple

from LogisticRegression.model import LinearModel
from utils import get_data_batch
import math


def calculate_loss(
        model: LinearModel, X: np.ndarray, y: np.ndarray, is_binary: bool = False
) -> float:
    '''
    Calculate the loss of the model on the given data.

    Args:
        model: LinearModel, the model to be evaluated
        X: np.ndarray, features
        y: np.ndarray, labels
    
    Returns:
        loss: float, loss of the model
    '''

    y_preds = model(X).squeeze()

    # Clip predicted probabilities to prevent values close to 0 or 1, 
    #which was otherwise resulting in -inf value after doing log
    #MODIFIED
    epsilon = 1e-15
    y_preds = np.clip(y_preds, epsilon, 1 - epsilon)
    

    if is_binary:
        loss = -np.mean(y * np.log(y_preds) + (1 - y) * np.log(1 - y_preds)) # binary cross-entropy loss
    else:
        #raise NotImplementedError('Calculate cross-entropy loss here')
        
        #y has shape (batch_size,) and y_preds has shape (batch_size,10) 
        #so create one_hot_encoding of y to represent it in shape (batch_size,10)
        batch_size=y_preds.shape[0]
        y_one_hot_encoding = np.zeros((batch_size, 10))
        y_one_hot_encoding[np.arange(batch_size), y] = 1
        loss = -np.mean(np.sum(y_one_hot_encoding * np.log(y_preds), axis=1))
    return loss


def calculate_accuracy(
        model: LinearModel, X: np.ndarray, y: np.ndarray, is_binary: bool = False
) -> float:
    '''
    Calculate the accuracy of the model on the given data.

    Args:
        model: LinearModel, the model to be evaluated
        X: np.ndarray, features
        y: np.ndarray, labels
    
    Returns:
        acc: float, accuracy of the model
    '''
    y_preds = model(X).squeeze()
    if is_binary:
        acc = np.mean((y_preds > 0.5) == y) # binary classification accuracy
    else:
        #raise NotImplementedError('Calculate accuracy for multi-class classification here')
        indx_with_max_probability = np.argmax(y_preds, axis=1)
        # if len(y)<20:
        #     print('in calc accuracy')
        #     print(f'y_preds={y_preds}')
        #     print(f'indx = {indx_with_max_probability}')
        #     print(f' y = {y}')
        acc = np.mean(indx_with_max_probability == y)
    return acc


def evaluate_model(
        model: LinearModel, X: np.ndarray, y: np.ndarray,
        batch_size: int, is_binary: bool = False
) -> Tuple[float, float]:
    '''
    Evaluate the model on the given data and return the loss and accuracy.

    Args:
        model: LinearModel, the model to be evaluated
        X: np.ndarray, features
        y: np.ndarray, labels
        batch_size: int, batch size for evaluation
    
    Returns:
        loss: float, loss of the model
        acc: float, accuracy of the model
    '''
    # get predicitions
    #raise NotImplementedError(
    #    'Get predictions in batches here (otherwise memory error for large datasets)')
    
    #num_batches = math.ceil(len(X) / batch_size )

    num_batches=(len(X) + batch_size - 1) // batch_size
    total_loss = 0.0
    total_acc = 0.0
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X))
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        y_preds_batch = model(X_batch)

        # Calculate loss for the batch
        batch_loss = calculate_loss(model, X_batch, y_batch, is_binary)
        total_loss += batch_loss * len(X_batch)

        # Calculate accuracy for the batch
        batch_acc = calculate_accuracy(model, X_batch, y_batch, is_binary)
        total_acc += batch_acc * len(X_batch)

    # Calculate loss and accuracy  -- MODIFIED
    loss = total_loss / len(X)
    acc = total_acc / len(X)

    return loss, acc

    


def fit_model(
        model: LinearModel, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, num_iters: int,
        lr: float, batch_size: int, l2_lambda: float,
        grad_norm_clip: float, is_binary: bool = False
) -> Tuple[list, list, list, list]:
    '''
    Fit the model on the given training data and return the training and validation
    losses and accuracies.

    Args:
        model: LinearModel, the model to be trained
        X_train: np.ndarray, features for training
        y_train: np.ndarray, labels for training
        X_val: np.ndarray, features for validation
        y_val: np.ndarray, labels for validation
        num_iters: int, number of iterations for training
        lr: float, learning rate for training
        batch_size: int, batch size for training
        l2_lambda: float, L2 regularization for training
        grad_norm_clip: float, clip value for gradient norm
        is_binary: bool, if True, use binary classification
    
    Returns:
        train_losses: list, training losses
        train_accs: list, training accuracies
        val_losses: list, validation losses
        val_accs: list, validation accuracies
    '''
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    continue_training=True
    number_of_violations =0
    val_loss_lowest =None

    # data_mean = np.mean(X_train, axis=0)
    # data_std_dev = np.std(X_train, axis=0)
    # if is_binary==False:
    #     X_train =(X_train - data_mean)/data_std_dev
    # val = 0

    for i in range(num_iters + 1):
        # if i%200 == 0:
        #     # val+=1
        #     lr *= 0.9
        # get batch
        X_batch, y_batch = get_data_batch(X_train, y_train, batch_size)
        
        # get predicitions
        # print("X_Batch shape: ", X_batch.shape)
        # print("X_Batch: ", X_batch)
        y_preds = model(X_batch).squeeze()
        
        # calculate loss
        loss = calculate_loss(model, X_batch, y_batch, is_binary)
        
        # calculate accuracy
        acc = calculate_accuracy(model, X_batch, y_batch, is_binary)
        
        # calculate gradient
        if is_binary:
            grad_W = ((y_preds - y_batch) @ X_batch).reshape(-1, 1)
            grad_b = np.mean(y_preds - y_batch)
        else:
            #y_batch has shape (batch_size,) and y_preds has shape (batch_size,10) 
            #so create one_hot_encoding of y to represent it in shape (batch_size,10)
            y_one_hot_encoding = np.zeros((batch_size, 10))
            y_one_hot_encoding[np.arange(batch_size), y_batch] = 1

            #grad_W = np.transpose((np.transpose(y_preds - y_one_hot_encoding) @ X_batch))#.reshape(-1, 1)
            grad_W = -np.dot(X_batch.T, (y_one_hot_encoding - y_preds)) / batch_size

        
            # if batch_size<16:
            #     print(grad_W)

            grad_b = np.mean(y_preds - y_one_hot_encoding, axis=0)
                   
        # regularization
        grad_W += l2_lambda * model.W
        grad_b += l2_lambda * model.b
        
        # clip gradient norm
        #raise NotImplementedError('Clip gradient norm here')
        grad_norm = np.sqrt(np.sum(grad_W ** 2) + grad_b ** 2)
        grad_norm_value = np.sqrt(np.sum(grad_norm ** 2))
        if grad_norm_value > grad_norm_clip:
            scaling_factor = grad_norm_clip / grad_norm_value
            grad_W = grad_W * scaling_factor
            grad_b = grad_b * scaling_factor
            

        # print(grad_W)
        # print(grad_b)

        # update model
        #raise NotImplementedError('Update model here (perform SGD)')
        model.W -= lr * grad_W
        model.b -= lr * grad_b

        if i % 10 == 0:
            # append loss
            train_losses.append(loss)
            # append accuracy
            train_accs.append(acc)

            # evaluate model
            val_loss, val_acc = evaluate_model(
                model, X_val, y_val, batch_size, is_binary)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(
                f'Iter {i}/{num_iters} - Train Loss: {loss:.4f} - Train Acc: {acc:.4f}'
                f' - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}'
            )
            
            # TODO: early stopping here if required
            # Stop when performance stops improving or starts degrading
            # this helps to prevent overfitting
            
            # if val_loss_lowest is None:
            #     val_loss_lowest = val_loss
            # elif val_loss <= val_loss_lowest:
            #     val_loss_lowest=val_loss
            #     number_of_violations =0
            # else:
            #     number_of_violations +=1

            # if number_of_violations>=5 and i>= num_iters/3:
            #     continue_training=False    

        if continue_training==False:
            break

    return train_losses, train_accs, val_losses, val_accs