import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

from datasets import load_metric
import torch
from transformers import TrainingArguments, AdapterTrainer, logging
from transformers.adapters import BertAdapterModel

class BERTAdapter:
    """
    A wrapper class for the BERT+Adapter implementation
    """
    
    def __init__(
        self,
        num_epochs,
        batch_size,
        learning_rate,
        category,
        device='cpu',
    ):
        """
        Initializes the BERT+Adapter model with given configurations.
        
        Args:
            num_epochs: Number of epochs to train the model.
            batch_size: Size of batches for training and evaluation.
            learning_rate: Learning rate for the optimizer.
            category: A string indicating the type of task (e.g., 'C' for classification).
            device: The device type ('cuda' or 'cpu') to run the training on. Defaults to 'cpu'.
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.category = category
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_labels = 2 if self.category == "C" else 1
        
        # # Set logging verbosity to error to reduce log clutter
        # logging.set_verbosity_error()

        # Load the model with the specified device
        self.model = BertAdapterModel.from_pretrained("bert-base-uncased", num_labels=self.num_labels).to(self.device)


    def fit(self, X, y, X_val, y_val):
        """
        Fits the BERT+Adapter model to the given training data.
        
        Args:
            X: Train set for a tokenized HuggingFace dataset
            y: A string indicating the target label name in the train dataset (e.g, 'Y').
            X_val: (Optional) Validation set for a tokenized HuggingFace dataset.
            y_val: (Optional) A string indicating the target label name in the validation dataset (e.g, 'Y').
         
        Returns:
            None
        """
        subset_train = self._prepare_dataset(dataset=X, label=y)
        
        # Check for validation data & setup training arguments
        if X_val is not None:
            subset_eval = self._prepare_dataset(dataset=X_val, label=y_val)
            training_args = self._setup_training_args(is_validation=True)
        else:
            training_args = self._setup_training_args()

        # Initialize the adapter and prediction head
        self.model.add_classification_head(self.category, num_labels=self.num_labels)
        self.model.add_adapter(self.category)
        self.model.train_adapter(self.category)
        self.model.set_active_adapters(self.category)

        # Define compute metrics for computing evaluation accuracy
        def compute_metrics(eval_pred):
            metric = load_metric('accuracy')
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)
        
        # Initialize trainer
        trainer = AdapterTrainer(model=self.model, 
                        args=training_args, 
                        train_dataset=subset_train, 
                        eval_dataset=subset_eval,
                        compute_metrics=compute_metrics)
        trainer.train()

    
    def predict(self, X):
        """
        Predicts labels for the given input data.
        
        Args:
            X: A list of indices representing the examples to predict on.
        
        Returns:
            A (numpy) array of predictions.
        """
        subset = self._prepare_dataset(X)
        dataloader = torch.utils.data.DataLoader(subset, batch_size=self.batch_size)

        preds = []
        self.model.eval()
        for batch in tqdm(dataloader, desc="Predicting"):
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs).logits
                if self.category == "C":
                    outputs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
                preds.append(outputs.detach().cpu().numpy())
        torch.cuda.empty_cache()
        return np.concatenate(preds).reshape(-1, 1)
    
    
    def predict_proba(self, X):
        """
        Predicts class probabilities for the given input data.
        
        Args:
            X: A list of indices representing the examples to predict on.
        
        Returns:
            An array of predicted class probabilities.
        """
        predictions = self.predict(X)
        difference = np.ones(predictions.shape) - predictions
        return np.concatenate((difference, predictions), axis=1)
    
    
    def _prepare_dataset(self, dataset, label=None):
        """
        Prepares a dataset for training or prediction.

        Args:
            dataset: Tokenized 'datasets.arrow_dataset.Dataset' with 'input_ids' and 'attention_mask' present.
            label: (Optional) A string indicating the target label name in the dataset (e.g, 'Y').

        Returns:
            A torch-formatted dataset ready for training or prediction.
        """
        columns_to_select = ["input_ids", "attention_mask"]
        if label is not None:
            dataset = dataset.rename_column(label, "labels")
            columns_to_select.append("labels")
        dataset.set_format(type="torch", columns=columns_to_select)
        return dataset
    

    def _setup_training_args(self, is_validation=False):
        """
        Sets up the training arguments for the adapter trainer.

        Returns:
            A TrainingArguments instance with the configured training parameters.
        """
        
        training_args = TrainingArguments(
            output_dir="output_dir",
            overwrite_output_dir=True,
            remove_unused_columns=False,
            dataloader_drop_last=False,
            save_total_limit=1,
            per_device_train_batch_size=self.batch_size,
            logging_strategy="epoch",
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs
        )
        if is_validation:
            training_args.per_device_eval_batch_size=self.batch_size
            training_args.evaluation_strategy="epoch"

        return training_args
    