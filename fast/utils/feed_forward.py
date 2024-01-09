import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import time
from .energy import get_energy


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

    def __init__(self, category, norm, input_size, layer_size, num_layers, num_classes):
        """
        Initializes a multilayer perceptron

        Args:
            category: determines the type of problem: "BC" for classification, "R" for regression.
            norm: if True, batch normalization will be applied.
            size: size of the input and hidden layer.
        """
        super(MLP, self).__init__()
        self.category = category
        modules = []

        if norm:
            for i in range(num_layers):
                if i == 0:  # for first layer
                    modules.append(nn.Linear(input_size, layer_size))
                    modules.append(nn.BatchNorm1d(layer_size))
                    modules.append(nn.ReLU())
                else:
                    modules.append(nn.Linear(layer_size, layer_size))
                    modules.append(nn.BatchNorm1d(layer_size))
                    modules.append(nn.ReLU())
        else:  # no normalization
            for i in range(num_layers):
                if i == 0:  # for first layer
                    modules.append(nn.Linear(input_size, layer_size))
                    modules.append(nn.ReLU())
                else:
                    modules.append(nn.Linear(layer_size, layer_size))
                    modules.append(nn.ReLU())

        self.linear_relu = nn.Sequential(*modules)
        self.linear = nn.Linear(layer_size, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: input tensor.

        Returns:
            network output (torch array)
        """
        out = self.linear_relu(x)
        y_logits = self.linear(out)
        return y_logits


class FeedForward:
    """
    A class that encapsulates the training and prediction for a feedforward neural network.
    """

    def __init__(
        self,
        max_epochs,
        batch_size,
        learning_rate,
        category,
        norm,
        input_size,
        layer_size,
        num_classes,
        num_layers,
        weight_decay,
        patience,
        min_delta,
        device,
        verbose=False
    ):
        """
        Initializes the FeedForward object with the given configurations.

        Args:
            max_epochs: max number of epochs to train.
            batch_size: size of the batch used in training.
            learning_rate: learning rate for the optimizer.
            category: type of problem ("BC" for binary classification, "R" for regression).
            norm: whether to use batch normalization.
            size: size of the input layer and the hidden layers.
            num_layers: number of ReLU hidden layers.
            patience: #epochs to wait for improvement in monitored metric before stopping the training.
            min_delta: minimum change in monitored metric that resets the patience counter.
            device: device to run the model on (e.g., 'cuda' or 'cpu'). Defaults to 'cpu'.
        """
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.category = category
        self.patience = patience
        self.min_delta = min_delta
        self.num_classes = num_classes
        self.verbose = verbose

        # Ensure device is set correctly depending on cuda availability
        self.device = torch.device(device)

        # Initialize the MLP model with the specified parameters
        self.model = MLP(category=category,
                         norm=norm,
                         input_size=input_size,
                         layer_size=layer_size,
                         num_layers=num_layers,
                         num_classes=num_classes).to(self.device)

        # Choose the appropriate loss function based on the problem category
        if category == "BC":
            self.loss_function = nn.BCEWithLogitsLoss()
        elif category == "MC":
            self.loss_function = nn.CrossEntropyLoss()
        elif category == "R":
            self.loss_function = nn.MSELoss()
        else:
            raise ValueError(
                "category must be either 'MC' for multi-class classification, 'BC' for binary classification or 'R' for regression.")

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Benchmarking: track training time and power usage per epoch
        self.train_times_per_epoch = []
        self.energy_per_epoch = []

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
            self.stopper = EarlyStopper(
                patience=self.patience, min_delta=self.min_delta)

        # Run the training loop
        trainloader = torch.utils.data.DataLoader(Data(
            X, y), batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)

        metrics_all = []
        is_stop = False
        for epoch in range(self.max_epochs):
            epoch_start_time = time.time()  # Start time for this epoch

            current_loss = []

            self.model.train()
            for data in trainloader:
                inputs, targets = data
                inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)
                targets = targets.reshape((targets.shape[0], self.num_classes))

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.loss_function(outputs, targets).to(self.device)

                loss.backward()
                current_loss.append(loss.item())

                self.optimizer.step()

            # Validation phase
            if X_val is not None and y_val is not None:
                metrics = self._validate(X_val, y_val)
                metrics["epoch"] = epoch + 1
                metrics_all.append(metrics)

                epoch_train_time = time.time() - epoch_start_time
                self.train_times_per_epoch.append(epoch_train_time)
                self.energy_per_epoch.append(
                    get_energy(epoch_train_time, self.device))
                if self.verbose:
                    print(
                        f"Epoch {metrics['epoch']}/{self.max_epochs} | Training Loss : {loss.item()} | Validation Loss : {metrics['loss']}")
                    print(
                        f"Accuracy : {metrics['accuracy']} | f1 : {metrics['f1']}\n")
                if self.stopper.early_stop(metrics["loss"]):
                    is_stop = True
                    break
            else:
                epoch_train_time = time.time() - epoch_start_time
                self.train_times_per_epoch.append(epoch_train_time)
                self.energy_per_epoch.append(
                    get_energy(epoch_train_time, self.device))
                # print(
                #     f"Epoch {metrics['epoch']}/{self.max_epochs} | Training Loss : {average_loss}")

        # print(f'Training process has finished.')
        if X_val is not None and y_val is not None:
            if is_stop:
                metrics = metrics_all[-1*self.patience - 1]
            else:
                metrics = metrics_all[-1]
        else:
            metrics = {}  # no metrics to report

        return metrics, self.train_times_per_epoch, self.energy_per_epoch

    def _validate(self, X, y_true, return_predictions=False):
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
        inputs_val = torch.from_numpy(X).float().to(self.device)
        targets_val = torch.from_numpy(y_true).float().to(
            self.device).reshape((-1, self.num_classes))

        with torch.no_grad():
            outputs_val = self.model(inputs_val)
            validation_loss = self.loss_function(
                outputs_val, targets_val).item()

        if self.category == "BC":
            y_pred = (outputs_val >= 0.5).to(int).cpu()
            metrics = {
                "loss": validation_loss,
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, average="micro"),
                "mcc": matthews_corrcoef(y_true, y_pred)
            }
        elif self.category == "MC":
            softmax = nn.Softmax(dim=1)
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(softmax(
                outputs_val.cpu()), axis=1)
            metrics = {
                "loss": validation_loss,
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, average="micro"),
                "mcc": matthews_corrcoef(y_true, y_pred)
            }
        elif self.category == "R":
            outputs_val = outputs_val.squeeze().cpu().numpy()
            metrics = {
                "loss": validation_loss,
                "pearson": pearsonr(y_true, outputs_val).statistic,
                "spearman": spearmanr(y_true, outputs_val).statistic,
            }

        if return_predictions:
            metrics["predictions"] = y_pred

        return metrics

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
        if self.category != "BC":
            raise ValueError(
                "predict_proba is only applicable to classification problems ('BC').")
        predictions = self.predict(X)
        difference = np.ones(predictions.shape) - predictions
        return np.concatenate((difference, predictions), axis=1)
