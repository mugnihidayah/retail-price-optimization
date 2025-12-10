"""
Unit Tests for PricingModel class
Tests model training, prediction, saving/loading, and hyperparameter tuning
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_training import PricingModel, train_and_compare_models


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample training data"""
    np.random.seed(42)
    n_samples = 100
    
    # Create features
    data = {
        'unit_price': np.random.uniform(10, 100, n_samples),
        'freight_price': np.random.uniform(1, 10, n_samples),
        'comp_1': np.random.uniform(10, 100, n_samples),
        'comp_2': np.random.uniform(10, 100, n_samples),
        'comp_3': np.random.uniform(10, 100, n_samples),
        'product_score': np.random.uniform(1, 5, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['diff_comp_1'] = df['unit_price'] - df['comp_1']
    df['diff_comp_2'] = df['unit_price'] - df['comp_2']
    df['diff_comp_3'] = df['unit_price'] - df['comp_3']
    df['ratio_comp_1'] = df['unit_price'] / df['comp_1']
    df['ratio_comp_2'] = df['unit_price'] / df['comp_2']
    df['ratio_comp_3'] = df['unit_price'] / df['comp_3']
    df['total_cost'] = df['unit_price'] + df['freight_price']
    
    # Create target (simulated demand based on price)
    df['qty'] = 100 - 0.5 * df['unit_price'] + np.random.normal(0, 5, n_samples)
    df['qty'] = np.maximum(df['qty'], 0)  # No negative quantities
    
    return df


@pytest.fixture
def feature_target_split(sample_data):
    """Split sample data into features and target"""
    X = sample_data.drop('qty', axis=1)
    y = sample_data['qty']
    return X, y


class TestPricingModelInit:
    """Tests for PricingModel initialization"""
    
    def test_default_init(self):
        """Test default initialization"""
        model = PricingModel()
        assert model.model_type == 'random_forest'
        assert model.random_state == 42
        assert model.model is not None
        assert model.feature_names is None
        assert model.is_tuned == False
    
    def test_custom_model_type(self):
        """Test initialization with different model types"""
        model = PricingModel(model_type='gradient_boosting')
        assert model.model_type == 'gradient_boosting'
    
    def test_invalid_model_type_fallback(self):
        """Test that invalid model type falls back to random_forest"""
        model = PricingModel(model_type='invalid_type')
        assert model.model_type == 'random_forest'
    
    def test_repr(self):
        """Test string representation"""
        model = PricingModel()
        assert 'PricingModel' in repr(model)
        assert 'random_forest' in repr(model)
        assert 'not fitted' in repr(model)


class TestPricingModelFit:
    """Tests for model fitting"""
    
    def test_fit_basic(self, feature_target_split):
        """Test basic model fitting"""
        X, y = feature_target_split
        model = PricingModel()
        model.fit(X, y, cv_folds=3, verbose=False)
        
        assert model.feature_names is not None
        assert len(model.feature_names) == len(X.columns)
        assert model.cv_scores is not None
        assert len(model.cv_scores) == 3
    
    def test_fit_stores_metrics(self, feature_target_split):
        """Test that fitting stores all required metrics"""
        X, y = feature_target_split
        model = PricingModel()
        model.fit(X, y, cv_folds=3, verbose=False)
        
        required_metrics = ['train_r2', 'train_mae', 'train_rmse', 'train_mape', 
                          'cv_r2_mean', 'cv_r2_std']
        for metric in required_metrics:
            assert metric in model.metrics, f"Missing metric: {metric}"
    
    def test_fit_method_chaining(self, feature_target_split):
        """Test that fit returns self for method chaining"""
        X, y = feature_target_split
        model = PricingModel()
        result = model.fit(X, y, cv_folds=3, verbose=False)
        
        assert result is model


class TestPricingModelPredict:
    """Tests for model prediction"""
    
    def test_predict_basic(self, feature_target_split):
        """Test basic prediction"""
        X, y = feature_target_split
        model = PricingModel()
        model.fit(X, y, cv_folds=3, verbose=False)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_unfitted_raises(self, feature_target_split):
        """Test that predicting with unfitted model raises error"""
        X, _ = feature_target_split
        model = PricingModel()
        model.model = None  # Force unfitted state
        
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X)
    
    def test_predict_with_uncertainty(self, feature_target_split):
        """Test prediction with uncertainty"""
        X, y = feature_target_split
        model = PricingModel()
        model.fit(X, y, cv_folds=3, verbose=False)
        
        predictions, std_devs = model.predict_with_uncertainty(X)
        
        assert len(predictions) == len(X)
        assert len(std_devs) == len(X)
        assert all(std >= 0 for std in std_devs)


class TestPricingModelEvaluation:
    """Tests for model evaluation methods"""
    
    def test_evaluate(self, feature_target_split):
        """Test evaluate method"""
        X, y = feature_target_split
        model = PricingModel()
        model.fit(X, y, cv_folds=3, verbose=False)
        
        metrics = model.evaluate(X, y)
        
        assert 'test_r2' in metrics
        assert 'test_mae' in metrics
        assert 'test_rmse' in metrics
        assert 'test_mape' in metrics
    
    def test_get_residuals(self, feature_target_split):
        """Test residual analysis"""
        X, y = feature_target_split
        model = PricingModel()
        model.fit(X, y, cv_folds=3, verbose=False)
        
        residuals_df = model.get_residuals(X, y)
        
        assert 'actual' in residuals_df.columns
        assert 'predicted' in residuals_df.columns
        assert 'residual' in residuals_df.columns
        assert 'abs_residual' in residuals_df.columns
        assert 'pct_error' in residuals_df.columns
        assert len(residuals_df) == len(X)
    
    def test_feature_importance(self, feature_target_split):
        """Test feature importance"""
        X, y = feature_target_split
        model = PricingModel()
        model.fit(X, y, cv_folds=3, verbose=False)
        
        importance = model.get_feature_importance()
        
        assert not importance.empty
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert 'cumulative' in importance.columns
        assert len(importance) == len(X.columns)


class TestPricingModelSaveLoad:
    """Tests for model saving and loading"""
    
    def test_save_load_roundtrip(self, feature_target_split):
        """Test that save/load preserves model state"""
        X, y = feature_target_split
        model = PricingModel()
        model.fit(X, y, cv_folds=3, verbose=False)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name
        
        try:
            # Save
            model.save(filepath)
            
            # Load
            loaded_model = PricingModel.load(filepath)
            
            # Compare
            assert loaded_model.model_type == model.model_type
            assert loaded_model.feature_names == model.feature_names
            assert loaded_model.metrics == model.metrics
            
            # Predictions should be identical
            orig_preds = model.predict(X)
            loaded_preds = loaded_model.predict(X)
            np.testing.assert_array_almost_equal(orig_preds, loaded_preds)
        finally:
            os.unlink(filepath)
    
    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            PricingModel.load('/nonexistent/path/model.pkl')


class TestPricingModelTuning:
    """Tests for hyperparameter tuning"""
    
    def test_tune_basic(self, feature_target_split):
        """Test basic hyperparameter tuning"""
        X, y = feature_target_split
        model = PricingModel()
        
        # Use small param grid for speed
        small_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        model.tune(X, y, param_grid=small_grid, cv_folds=2, verbose=False)
        
        assert model.is_tuned == True
        assert model.best_params is not None
    
    def test_tune_then_fit(self, feature_target_split):
        """Test tuning followed by fitting"""
        X, y = feature_target_split
        model = PricingModel()
        
        small_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        model.tune(X, y, param_grid=small_grid, cv_folds=2, verbose=False)
        model.fit(X, y, cv_folds=3, verbose=False)
        
        assert model.is_tuned == True
        assert 'train_r2' in model.metrics


class TestTrainAndCompare:
    """Tests for train_and_compare_models function"""
    
    def test_compare_models(self, feature_target_split):
        """Test comparing multiple models"""
        X, y = feature_target_split
        
        models = train_and_compare_models(X, y, cv_folds=2, tune_models=False)
        
        assert 'random_forest' in models
        assert 'gradient_boosting' in models
        
        for name, model in models.items():
            assert isinstance(model, PricingModel)
            assert model.feature_names is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
