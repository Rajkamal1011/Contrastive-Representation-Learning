import torch
from argparse import Namespace

import ContrastiveRepresentation.pytorch_utils as ptu
from utils import *
from LogisticRegression.model import SoftmaxRegression as LinearClassifier
from ContrastiveRepresentation.model import Encoder, Classifier
from ContrastiveRepresentation.train_utils import fit_contrastive_model, fit_model


def main(args: Namespace):
    '''
    Main function to train and generate predictions in csv format

    Args:
    - args : Namespace : command line arguments
    '''
    # Set the seed
    torch.manual_seed(args.sr_no)

    # Get the training data
    X, y = get_data(args.train_data_path)
    X_train, y_train, X_val, y_val = train_test_split(X, y)
    # print("TYPE: ",X_train, y_train, X_val, y_val)

    # Convert the images and labels to torch tensors
    X_train, y_train = ptu.from_numpy(X_train).float(), ptu.from_numpy(y_train).long()
    X_val, y_val = ptu.from_numpy(X_val).float(), ptu.from_numpy(y_val).long()
    # TODO: Convert the images and labels to torch tensors using pytorch utils (ptu)
    
    # Create the model
    encoder = Encoder(args.z_dim).to(ptu.device)
    if args.mode == 'fine_tune_linear':
        # classifier = # TODO: Create the linear classifier model
        classifier = LinearClassifier(args.z_dim, 10)
        # classifier = classifier.to(ptu.device)
    elif args.mode == 'fine_tune_nn':
        # classifier = # TODO: Create the neutrain_embeddings_npral network classifier model
        classifier = Classifier(args.z_dim, 10)
        classifier = classifier.to(ptu.device)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'cont_rep':
        # raise NotImplementedError('Implement the contrastive representation learning')
        fit_contrastive_model(encoder, X_train, y_train, args.num_iters, args.batch_size, args.lr)

        encoder.eval()

        z_list = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for i in range(0, len(X_val), args.batch_size):
                X_batch = X_val[i:i+args.batch_size]
                X_batch = torch.tensor(X_batch, dtype=torch.float32)
                X_batch = X_batch.to(device)
                z_batch = encoder(X_batch)
                z_list.append(z_batch)
        
        z = torch.cat(z_list, dim = 0)
        # Plot the t-SNE after fitting the encoder
        plot_tsne(z, y_val)

        # save the encoder
        torch.save(encoder.state_dict(), 'encoder_state.pth')
    else: # train the classifier (fine-tune the encoder also when using NN classifier)
        # load the encoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.load_state_dict(torch.load('encoder_state.pth'))
                        
        print("LESS GO")
        # encoder = encoder.to(ptu.device)

        # raise NotImplementedError('Load the encoder')
        
        # Fit the model
        train_losses, train_accs, test_losses, test_accs = fit_model(
            encoder, classifier, X_train, y_train, X_val, y_val, args)
        print("MORE GO")
        # Plot the losses
        plot_losses(train_losses, test_losses, f'{args.mode} - Losses')
        
        # Plot the accuracies
        plot_accuracies(train_accs, test_accs, f'{args.mode} - Accuracies')
        
        # Get the test data
        X_test, _ = get_data(args.test_data_path)
        X_test = ptu.from_numpy(X_test).float()

        # Save the predictions for the test data in a CSV file
        y_preds = []
        for i in range(0, len(X_test), args.batch_size):
            X_batch = X_test[i:i+args.batch_size].to(ptu.device)
            repr_batch = encoder(X_batch)
            if 'linear' in args.mode:
                repr_batch = ptu.to_numpy(repr_batch)
            y_pred_batch = classifier(repr_batch)
            if 'nn' in args.mode:
                y_pred_batch = ptu.to_numpy(y_pred_batch)
            y_preds.append(y_pred_batch)
        y_preds = np.concatenate(y_preds).argmax(axis=1)
        np.savetxt(f'data/{args.sr_no}_{"repr_lin" if "linear" in args.mode else "repr_nn"}.csv',\
                y_preds, delimiter=',', fmt='%d')
        print(f'Predictions saved to data/{args.sr_no}_{"repr_lin" if "linear" in args.mode else "repr_nn"}.csv')
