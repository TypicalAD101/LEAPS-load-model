from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import pickle
from utils.logger import get_logger


class BaseTester(ABC):
    """
    Base class for model testing.
    Provides common functionality for loading models and calculating metrics.
    """
    
    def __init__(self, config_handler):
        self.config_handler = config_handler
        self.config = config_handler.get_config()
        self.logger = get_logger()
        self.model = None
        self.test_predictions = None
        self.test_actuals = None
        
    @abstractmethod
    def load_model(self, model_path):
        """
        Load the trained model from the specified path.
        Must be implemented by each model-specific tester.
        """
        pass
    
    @abstractmethod
    def prepare_test_data(self, test_data):
        """
        Prepare test data for prediction.
        Must be implemented by each model-specific tester.
        """
        pass
    
    @abstractmethod
    def predict(self, test_data):
        """
        Make predictions on test data.
        Must be implemented by each model-specific tester.
        """
        pass
    
    def calculate_metrics(self, predictions, actuals):
        """
        Calculate common evaluation metrics.
        """
        try:
            # Ensure we have numpy arrays
            predictions = np.array(predictions).flatten()
            actuals = np.array(actuals).flatten()
            
            # Calculate metrics
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            r2 = r2_score(actuals, predictions)
            
            # Additional metrics
            mape = np.mean(np.abs((actuals - predictions) / np.where(actuals != 0, actuals, 1))) * 100
            
            metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return None
    
    def print_test_results(self, metrics, predictions, actuals):
        """
        Print test results and summary statistics.
        """
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        
        if metrics:
            print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
            print(f"Root Mean Square Error (RMSE): {metrics['RMSE']:.4f}")
            print(f"R-squared (R²): {metrics['R2']:.4f}")
            print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
        
        # Print sample predictions
        print(f"\nSample Predictions (first 10):")
        print("Actual\t\tPredicted\t\tDifference")
        print("-" * 50)
        
        # Debug: Print first few values to understand structure
        if len(actuals) > 0 and len(predictions) > 0:
            print(f"DEBUG - First actual type: {type(actuals[0])}, value: {actuals[0]}")
            print(f"DEBUG - First prediction type: {type(predictions[0])}, value: {predictions[0]}")
        
        for i in range(min(10, len(predictions))):
            actual = actuals[i] if i < len(actuals) else "N/A"
            pred = predictions[i] if i < len(predictions) else "N/A"
            
            # Convert to scalar values
            try:
                if hasattr(actual, '__iter__') and not isinstance(actual, str):
                    actual = float(actual[0]) if len(actual) > 0 else "N/A"
                else:
                    actual = float(actual)
                
                if hasattr(pred, '__iter__') and not isinstance(pred, str):
                    pred = float(pred[0]) if len(pred) > 0 else "N/A"
                else:
                    pred = float(pred)
                
                if isinstance(actual, (int, float)) and isinstance(pred, (int, float)):
                    diff = abs(actual - pred)
                    print(f"{actual:.4f}\t\t{pred:.4f}\t\t{diff:.4f}")
                else:
                    print(f"{actual}\t\t{pred}\t\tN/A")
            except (ValueError, TypeError, IndexError):
                print(f"{actual}\t\t{pred}\t\tN/A")
        
        print("\n" + "="*60)
    
    def test(self, test_data, model_path):
        """
        Main testing method that orchestrates the testing process.
        """
        try:
            self.logger.info("Starting model testing...")
            
            # Load the model
            self.logger.info(f"Loading model from {model_path}")
            self.load_model(model_path)
            
            if self.model is None:
                raise ValueError("Failed to load model")
            
            # Prepare test data
            self.logger.info("Preparing test data...")
            prepared_data = self.prepare_test_data(test_data)
            
            # Make predictions
            self.logger.info("Making predictions...")
            predictions = self.predict(prepared_data)
            
            # Extract actual values (assuming test_data contains targets)
            if hasattr(test_data, 'targets'):
                actuals = test_data.targets
            elif isinstance(test_data, (list, tuple)) and len(test_data) > 1:
                actuals = test_data[1]  # Assuming (features, targets) format
            else:
                # Try to extract targets from the data structure
                actuals = self.extract_targets(test_data)
            
            # Calculate metrics
            self.logger.info("Calculating metrics...")
            metrics = self.calculate_metrics(predictions, actuals)
            
            # Print results
            self.print_test_results(metrics, predictions, actuals)
            
            # Store results for potential further use
            self.test_predictions = predictions
            self.test_actuals = actuals
            
            self.logger.info("Testing completed successfully!")
            return metrics, predictions, actuals
            
        except Exception as e:
            self.logger.error(f"Error during testing: {str(e)}")
            raise
    
    def extract_targets(self, test_data):
        """
        Extract target values from test data.
        Override this method if needed for specific data formats.
        """
        # Default implementation - may need to be overridden
        if hasattr(test_data, 'y'):
            return test_data.y
        elif hasattr(test_data, 'target'):
            return test_data.target
        elif hasattr(test_data, 'targets'):
            return test_data.targets
        else:
            raise ValueError("Could not extract targets from test data. Override extract_targets method.") 