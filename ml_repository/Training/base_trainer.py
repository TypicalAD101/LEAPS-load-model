from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """
    Abstract base class for all model trainers.
    Enforces a consistent interface for training models.
    """
    @abstractmethod
    def train(self, *args, model_save_path: str, **kwargs):
        """
        Train the model and save it to the specified path.
        Args:
            *args: Model/data-specific arguments.
            model_save_path (str): Path to save the trained model.
            **kwargs: Additional keyword arguments.
        """
        pass 