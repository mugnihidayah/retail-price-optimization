"""
Model Training Module for Price Optimization
Includes XGBoost, LightGBM, and Random Forest with cross-validation,
hyperparameter tuning, and comprehensive evaluation metrics.
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

from sklearn.model_selection import (
    cross_val_score, train_test_split, KFold, 
    GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Local imports
try:
    from utils import calculate_mape, calculate_price_elasticity
except ImportError:
    # Fallback if utils not available
    def calculate_mape(y_true, y_pred):
        mask = y_true != 0
        if not mask.any():
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def calculate_price_elasticity(prices, quantities, method='point'):
        return np.zeros(len(prices))

# Try importing optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PricingModel:
    """
    Unified pricing model class supporting multiple algorithms
    with cross-validation, hyperparameter tuning, and confidence metrics.
    """
    
    # Default hyperparameter grids for tuning
    PARAM_GRIDS = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [8, 12, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9]
        },
        'lightgbm': {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [31, 50, 70]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9]
        }
    }
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize pricing model.
        
        Args:
            model_type: 'random_forest', 'xgboost', 'lightgbm', or 'gradient_boosting'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.cv_scores: Optional[np.ndarray] = None
        self.metrics: Dict[str, float] = {}
        self.best_params: Optional[Dict[str, Any]] = None
        self.is_tuned: bool = False
        
        self._init_model()
    
    def _init_model(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the underlying model based on model_type.
        
        Args:
            params: Optional custom parameters for the model
        """
        default_params = {
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        if self.model_type == 'random_forest':
            rf_params = {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                **default_params
            }
            self.model = RandomForestRegressor(**rf_params)
            
        elif self.model_type == 'xgboost' and HAS_XGBOOST:
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                **default_params
            }
            self.model = xgb.XGBRegressor(**xgb_params)
            
        elif self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            lgb_params = {
                'n_estimators': 100,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbose': -1,
                **default_params
            }
            self.model = lgb.LGBMRegressor(**lgb_params)
            
        elif self.model_type == 'gradient_boosting':
            gb_params = {
                'n_estimators': 100,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': self.random_state
                # Note: GradientBoostingRegressor doesn't support n_jobs
            }
            self.model = GradientBoostingRegressor(**gb_params)
            
        else:
            logger.warning(
                f"Model type '{self.model_type}' not available. "
                "Falling back to Random Forest."
            )
            self.model_type = 'random_forest'
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=self.random_state,
                n_jobs=-1
            )
    
    def tune(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        param_grid: Optional[Dict[str, List]] = None,
        cv_folds: int = 3,
        method: str = 'grid',
        n_iter: int = 20,
        verbose: bool = True
    ) -> 'PricingModel':
        """
        Tune hyperparameters using grid search or randomized search.
        
        Args:
            X: Feature dataframe
            y: Target series
            param_grid: Custom parameter grid (uses default if None)
            cv_folds: Number of cross-validation folds
            method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
            n_iter: Number of iterations for randomized search
            verbose: Whether to print progress
            
        Returns:
            Self for method chaining
        """
        if param_grid is None:
            param_grid = self.PARAM_GRIDS.get(self.model_type, {})
        
        if not param_grid:
            logger.warning("No parameter grid available for tuning.")
            return self
        
        if verbose:
            logger.info(f"Starting hyperparameter tuning for {self.model_type}...")
            logger.info(f"Method: {method}, CV folds: {cv_folds}")
        
        # Create search object
        if method == 'grid':
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1,
                verbose=1 if verbose else 0
            )
        else:
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1 if verbose else 0
            )
        
        # Fit search
        search.fit(X, y)
        
        # Store results
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        self.is_tuned = True
        
        if verbose:
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        return self
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        cv_folds: int = 5, 
        verbose: bool = True
    ) -> 'PricingModel':
        """
        Fit the model with cross-validation.
        
        Args:
            X: Feature dataframe
            y: Target series
            cv_folds: Number of cross-validation folds
            verbose: Whether to print progress
            
        Returns:
            Self for method chaining
        """
        self.feature_names = list(X.columns)
        
        # Cross-validation
        if verbose:
            logger.info(f"Running {cv_folds}-fold cross-validation...")
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        self.cv_scores = cross_val_score(self.model, X, y, cv=kfold, scoring='r2')
        
        if verbose:
            logger.info(f"CV R² scores: {self.cv_scores}")
            logger.info(f"Mean CV R²: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std()*2:.4f})")
        
        # Fit on full data
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        self.metrics = {
            'train_r2': r2_score(y, y_pred),
            'train_mae': mean_absolute_error(y, y_pred),
            'train_rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'train_mape': calculate_mape(y.values, y_pred),
            'cv_r2_mean': self.cv_scores.mean(),
            'cv_r2_std': self.cv_scores.std()
        }
        
        if verbose:
            logger.info(f"Training R²: {self.metrics['train_r2']:.4f}")
            logger.info(f"Training MAE: {self.metrics['train_mae']:.4f}")
            logger.info(f"Training RMSE: {self.metrics['train_rmse']:.4f}")
            logger.info(f"Training MAPE: {self.metrics['train_mape']:.2f}%")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of predictions
            
        Raises:
            ValueError: If model is not fitted
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Ensure correct feature order
        if self.feature_names:
            X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def predict_with_uncertainty(
        self, 
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        Only works for ensemble models (Random Forest, XGBoost, LightGBM).
        
        Args:
            X: Feature dataframe
            
        Returns:
            Tuple of (predictions, standard deviations)
        """
        if self.feature_names:
            X = X[self.feature_names]
        
        predictions = self.model.predict(X)
        
        # For Random Forest, we can get predictions from each tree
        if hasattr(self.model, 'estimators_'):
            # Get predictions from all trees
            all_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
            std_dev = np.std(all_preds, axis=0)
        else:
            # Use CV standard deviation as proxy
            cv_std = self.metrics.get('cv_r2_std', 0.1)
            std_dev = np.full(len(predictions), cv_std * np.std(predictions))
        
        return predictions, std_dev
    
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X: Feature dataframe
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        
        return {
            'test_r2': r2_score(y, y_pred),
            'test_mae': mean_absolute_error(y, y_pred),
            'test_rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'test_mape': calculate_mape(y.values, y_pred)
        }
    
    def get_residuals(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Get residual analysis dataframe.
        
        Args:
            X: Feature dataframe
            y: True target values
            
        Returns:
            DataFrame with predictions, residuals, and analysis
        """
        y_pred = self.predict(X)
        residuals = y.values - y_pred
        
        return pd.DataFrame({
            'actual': y.values,
            'predicted': y_pred,
            'residual': residuals,
            'abs_residual': np.abs(residuals),
            'pct_error': np.where(y.values != 0, residuals / y.values * 100, 0)
        })
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None or self.feature_names is None:
            return pd.DataFrame()
        
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Add cumulative importance
            importance['cumulative'] = importance['importance'].cumsum()
            
            return importance
        
        return pd.DataFrame()
    
    def save(self, filepath: str) -> None:
        """
        Save model and metadata.
        
        Args:
            filepath: Path to save the model
        """
        save_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'cv_scores': self.cv_scores,
            'metrics': self.metrics,
            'best_params': self.best_params,
            'is_tuned': self.is_tuned
        }
        joblib.dump(save_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PricingModel':
        """
        Load model from file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded PricingModel instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is corrupted
        """
        try:
            save_data = joblib.load(filepath)
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {e}")
            raise ValueError(f"Model file corrupted or incompatible: {e}")
        
        instance = cls(model_type=save_data.get('model_type', 'random_forest'))
        instance.model = save_data['model']
        instance.feature_names = save_data.get('feature_names')
        instance.cv_scores = save_data.get('cv_scores')
        instance.metrics = save_data.get('metrics', {})
        instance.best_params = save_data.get('best_params')
        instance.is_tuned = save_data.get('is_tuned', False)
        
        # Backward compatibility: if loading old model format
        if instance.feature_names is None and hasattr(instance.model, 'feature_names_in_'):
            instance.feature_names = list(instance.model.feature_names_in_)
        
        return instance
    
    @property
    def feature_names_in_(self) -> Optional[List[str]]:
        """Compatibility property for sklearn-style access."""
        return self.feature_names
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "tuned" if self.is_tuned else "default"
        fitted = "fitted" if self.feature_names else "not fitted"
        return f"PricingModel(type='{self.model_type}', {status}, {fitted})"


def train_and_compare_models(
    X: pd.DataFrame, 
    y: pd.Series, 
    cv_folds: int = 5,
    tune_models: bool = False
) -> Dict[str, PricingModel]:
    """
    Train multiple models and compare their performance.
    
    Args:
        X: Feature dataframe
        y: Target series
        cv_folds: Number of cross-validation folds
        tune_models: Whether to run hyperparameter tuning
    
    Returns:
        Dictionary of model_name -> fitted PricingModel
    """
    models = {}
    results = []
    
    model_types = ['random_forest', 'gradient_boosting']
    
    if HAS_XGBOOST:
        model_types.append('xgboost')
    if HAS_LIGHTGBM:
        model_types.append('lightgbm')
    
    for model_type in model_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_type}...")
        logger.info('='*50)
        
        model = PricingModel(model_type=model_type)
        
        # Optional hyperparameter tuning
        if tune_models:
            logger.info("Running hyperparameter tuning...")
            model.tune(X, y, cv_folds=3, method='random', n_iter=10)
        
        model.fit(X, y, cv_folds=cv_folds)
        models[model_type] = model
        
        results.append({
            'model': model_type,
            'tuned': model.is_tuned,
            'cv_r2_mean': model.metrics['cv_r2_mean'],
            'cv_r2_std': model.metrics['cv_r2_std'],
            'train_r2': model.metrics['train_r2'],
            'train_mae': model.metrics['train_mae'],
            'train_rmse': model.metrics['train_rmse'],
            'train_mape': model.metrics['train_mape']
        })
    
    # Print comparison
    results_df = pd.DataFrame(results)
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    print(results_df.to_string(index=False))
    
    # Find best model
    best_idx = results_df['cv_r2_mean'].idxmax()
    best_model = results_df.loc[best_idx, 'model']
    logger.info(
        f"\n🏆 Best model: {best_model} "
        f"(CV R² = {results_df.loc[best_idx, 'cv_r2_mean']:.4f})"
    )
    
    return models


if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    
    try:
        df = pd.read_csv('data/processed/train_data.csv')
    except FileNotFoundError:
        print("Error: Data file not found at 'data/processed/train_data.csv'")
        print("Please ensure the data file exists.")
        exit(1)
    
    # Prepare features and target
    target_col = 'qty'
    feature_cols = [c for c in df.columns if c != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
    
    # Train and compare models (with optional tuning)
    import argparse
    parser = argparse.ArgumentParser(description='Train pricing models')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    args = parser.parse_args()
    
    models = train_and_compare_models(X, y, cv_folds=5, tune_models=args.tune)
    
    # Save best model
    best_model_name = max(models.keys(), key=lambda k: models[k].metrics['cv_r2_mean'])
    best_model = models[best_model_name]
    
    # Create models directory if it doesn't exist
    Path('models').mkdir(exist_ok=True)
    
    best_model.save('models/pricing_model_enhanced.pkl')
    print(f"\n✅ Best model ({best_model_name}) saved to models/pricing_model_enhanced.pkl")
    
    # Print feature importance
    print("\nTop 10 Most Important Features:")
    importance = best_model.get_feature_importance()
    if not importance.empty:
        print(importance.head(10).to_string(index=False))
