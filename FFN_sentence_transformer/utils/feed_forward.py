import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class EarlyStopper:
    """
    A utility class that stops training when a monitored metric stops improving.
    """
    def __init__(self, patience, min_delta):
        """
        Initializes EarlyStopped class

        Args:
            patience: number of epochs to wait after min has been hit. After this number, training stops.
            min_delta: minimum change to qualify as an improvement. Smaller changes are ignored.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Checks whether the early stopping conditions are met.

        Args:
            validation_loss: the loss obtained after a validation epoch.
        
        Returns:
            True if training should be stopped, otherwise False.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Data(torch.utils.data.Dataset):
    """
    Dataset class that wraps the input and target tensors for the dataset.
    """
    def __init__(self, X, y):
        """
        X: features data
        y: target/output data
        """
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, i):
        """
        Retrieves the ith sample from the dataset.
        """
        return self.X[i], self.y[i]
    

class MLP(nn.Module):
    """
    A Multilayer Perceptron (MLP) neural network architecture for classification or regression.
    """
    def __init__(self, category, norm, size, num_layers):
        """
        Initializes a multilayer perceptron

        Args:
            category: determines the type of problem: "C" for classification, "R" for regression.
            norm: if True, batch normalization will be applied.
            size: size of the input and hidden layer.
        """
        super(MLP, self).__init__()
        self.category = category
        modules = []

        if norm:
            for _ in range(num_layers):
                modules.append(nn.Linear(size, size))
                modules.append(nn.BatchNorm1d(size))
                modules.append(nn.ReLU())
        else:
            for _ in range(num_layers):
                modules.append(nn.Linear(size, size))
                modules.append(nn.ReLU())

        self.linear_relu = nn.Sequential(*modules)
        self.linear = nn.Linear(size, 1)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: input tensor.
        
        Returns:
            network output (torch array)
        """
        out = self.linear_relu(x)
        y_pred = self.linear(out)
        if self.category == "C":
            return torch.sigmoid(y_pred)
        elif self.category == "R":
            return y_pred

class FeedForward:
    """
    A class that encapsulates the training and prediction for a feedforward neural network.
    """
    def __init__(
        self,
        num_epochs,
        batch_size,
        learning_rate,
        category,
        norm,
        size,
        num_layers,
        weight_decay,
        patience,
        min_delta,
        device='cpu'
    ):
        """
        Initializes the FeedForward object with the given configurations.

        Args:
            num_epochs: number of epochs to train.
            batch_size: size of the batch used in training.
            learning_rate: learning rate for the optimizer.
            category: type of problem ("C" for classification, "R" for regression).
            norm: whether to use batch normalization.
            size: size of the input layer and the hidden layers.
            num_layers: number of ReLU hidden layers.
            patience: #epochs to wait for improvement in monitored metric before stopping the training.
            min_delta: minimum change in monitored metric that resets the patience counter.
            device: device to run the model on (e.g., 'cuda' or 'cpu'). Defaults to 'cpu'.
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.category = category
        self.patience = patience
        self.min_delta = min_delta

        # Ensure device is set correctly depending on cuda availability
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize the MLP model with the specified parameters
        self.model = MLP(category=category, norm=norm, size=size, num_layers=num_layers).to(self.device)
        
        # Choose the appropriate loss function based on the problem category
        if category == "C":
            self.loss_function = nn.BCELoss()
        elif category == "R":
            self.loss_function = nn.MSELoss()
        
        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the model using the provided dataset.
        
        Args:
            X: feature matrix for training.
            y: target vector for training.
            X_val: feature matrix for validation (optional).
            y_val: target vector for validation (optional).
        
        Returns:
            tuple of the lowest validation loss and the best epoch number if validation is provided, otherwise None
        """
        # Define early stopper if validation data is provided
        if X_val is not None:
            self.stopper = EarlyStopper(patience=self.patience, min_delta=self.min_delta)

        # Run the training loop
        for epoch in range(self.num_epochs):

            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            self.model.train()
            trainloader = torch.utils.data.DataLoader(Data(X, y), batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=False)
            for _, data in enumerate(trainloader, 0):

                # Get and prepare inputs
                inputs, targets = data
                inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)
                targets = targets.reshape((targets.shape[0], 1))

                # Zero the gradients
                self.optimizer.zero_grad()

                # Perform forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.loss_function(outputs, targets).to(self.device)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                self.optimizer.step()
                current_loss += loss.item()
            # print(f"Epoch : {epoch+1}/{self.num_epochs} | Training Loss : {current_loss}")
            
            # Validation phase
            if X_val is not None and y_val is not None:
                validation_loss, validation_acc = self._validate(X_val, y_val)
                # print(f"Validation Loss : {validation_loss} | Validation Accuracy : {validation_acc}")
                if self.stopper.early_stop(validation_loss):
                    # print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

        # print(f'Training process has finished.')
        if X_val is not None and y_val is not None:
            return (epoch+1, validation_loss, validation_acc)

    def _validate(self, X_val, y_val):
        """
        Validates the model on a provided validation set.

        Args:
            X_val: feature matrix for validation.
            y_val: target vector for validation.

        Returns:
            validation loss and accuracy (for classification)
            validation loss (for regression)
        """
        self.model.eval()
        inputs_val = torch.from_numpy(X_val).float().to(self.device)
        targets_val = torch.from_numpy(y_val).float().to(self.device).reshape((-1, 1))
        
        with torch.no_grad():
            outputs_val = self.model(inputs_val)
            validation_loss = self.loss_function(outputs_val, targets_val).item()
            
        if self.category == "C":
            validation_acc = accuracy_score(y_val, (outputs_val >= 0.5).to(int).cpu())
            return validation_loss, validation_acc
        elif self.category == "R":
            return validation_loss, None

    def predict(self, X):
        """
        Predicts outputs for the given input X using the trained model.

        Args:
            X: input feature matrix.

        Returns:
            numpy array of predictions
        """
        # Put inputs on device
        inputs = torch.from_numpy(X).float().to(self.device)

        # Set the model to evaluation mode
        self.model.eval()

        # Disable gradient computation
        with torch.no_grad():
            # Forward pass to compute predictions
            predictions = self.model(inputs)

        # Move predictions back to the CPU and convert to numpy array
        predictions = predictions.cpu().numpy()
        return predictions

    def predict_proba(self, X):
        """
        Predicts class probabilities for each input sample for classification problems.

        Args:
            X: input feature matrix.

        Returns:
            numpy array of predicted class probabilities.
        """
        if self.category != "C":
            raise ValueError("predict_proba is only applicable to classification problems ('C').")
        predictions = self.predict(X)
        difference = np.ones(predictions.shape) - predictions
        return np.concatenate((difference, predictions), axis=1)
