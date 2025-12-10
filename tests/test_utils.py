"""
Unit Tests for Price Optimization System
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    safe_division, 
    calculate_competitor_features,
    calculate_cost_features,
    clip_predictions,
    format_currency,
    format_percentage,
    format_change,
    validate_input_data,
    calculate_revenue,
    calculate_profit,
    calculate_margin,
    calculate_mape,
    calculate_price_elasticity,
    classify_elasticity,
    validate_price_range,
    validate_product_data,
    validate_optimization_params,
    ValidationResult
)


class TestSafeDivision:
    """Tests for safe_division function"""
    
    def test_normal_division(self):
        """Test normal division works correctly"""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([2, 4, 5])
        result = safe_division(numerator, denominator)
        expected = np.array([5, 5, 6])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_division_by_zero(self):
        """Test that division by zero returns default value"""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([0, 4, 0])
        result = safe_division(numerator, denominator, default=1.0)
        expected = np.array([1.0, 5.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_custom_default(self):
        """Test custom default value for division by zero"""
        numerator = pd.Series([10])
        denominator = pd.Series([0])
        result = safe_division(numerator, denominator, default=999)
        assert result[0] == 999
    
    def test_numpy_array_input(self):
        """Test with numpy array inputs"""
        numerator = np.array([10, 20, 30])
        denominator = np.array([2, 0, 5])
        result = safe_division(numerator, denominator, default=0)
        expected = np.array([5, 0, 6])
        np.testing.assert_array_almost_equal(result, expected)


class TestCalculateCompetitorFeatures:
    """Tests for calculate_competitor_features function"""
    
    def test_competitor_features_created(self):
        """Test that competitor features are created correctly"""
        df = pd.DataFrame({
            'unit_price': [100],
            'comp_1': [90],
            'comp_2': [110],
            'comp_3': [100]
        })
        
        result = calculate_competitor_features(df)
        
        assert 'diff_comp_1' in result.columns
        assert 'ratio_comp_1' in result.columns
        assert result['diff_comp_1'].iloc[0] == 10  # 100 - 90
        assert result['ratio_comp_1'].iloc[0] == pytest.approx(100/90)
    
    def test_handles_zero_competitor_price(self):
        """Test that zero competitor price doesn't cause errors"""
        df = pd.DataFrame({
            'unit_price': [100],
            'comp_1': [0],  # Zero competitor price
            'comp_2': [110],
            'comp_3': [100]
        })
        
        result = calculate_competitor_features(df)
        
        # Should return default ratio of 1.0 for zero division
        assert result['ratio_comp_1'].iloc[0] == 1.0


class TestCalculateCostFeatures:
    """Tests for calculate_cost_features function"""
    
    def test_total_cost_calculated(self):
        """Test total cost is calculated correctly"""
        df = pd.DataFrame({
            'unit_price': [100, 200],
            'freight_price': [15, 25]
        })
        
        result = calculate_cost_features(df)
        
        assert 'total_cost' in result.columns
        assert result['total_cost'].iloc[0] == 115
        assert result['total_cost'].iloc[1] == 225


class TestClipPredictions:
    """Tests for clip_predictions function"""
    
    def test_clips_negative_values(self):
        """Test that negative values are clipped to min_val"""
        predictions = np.array([-5, 10, -3, 20])
        result = clip_predictions(predictions, min_val=0)
        expected = np.array([0, 10, 0, 20])
        np.testing.assert_array_equal(result, expected)
    
    def test_custom_min_val(self):
        """Test custom minimum value"""
        predictions = np.array([-5, 10, 3])
        result = clip_predictions(predictions, min_val=5)
        expected = np.array([5, 10, 5])
        np.testing.assert_array_equal(result, expected)
    
    def test_max_val_clipping(self):
        """Test maximum value clipping"""
        predictions = np.array([5, 10, 100])
        result = clip_predictions(predictions, min_val=0, max_val=50)
        expected = np.array([5, 10, 50])
        np.testing.assert_array_equal(result, expected)


class TestFormatFunctions:
    """Tests for formatting functions"""
    
    def test_format_currency(self):
        """Test currency formatting"""
        assert format_currency(100.0) == "$100.00"
        assert format_currency(1234.56) == "$1,234.56"
        assert format_currency(100.0, "€") == "€100.00"
    
    def test_format_percentage(self):
        """Test percentage formatting"""
        assert format_percentage(15.5) == "15.5%"
        assert format_percentage(15.567, decimals=2) == "15.57%"
    
    def test_format_change(self):
        """Test change formatting with sign"""
        assert format_change(15.5) == "+15.5%"
        assert format_change(-10.2) == "-10.2%"
        assert format_change(0) == "+0.0%"


class TestValidateInputData:
    """Tests for validate_input_data function"""
    
    def test_all_columns_present(self):
        """Test when all required columns are present"""
        df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        is_valid, missing = validate_input_data(df, ['a', 'b'])
        assert is_valid == True
        assert missing == []
    
    def test_missing_columns(self):
        """Test when some columns are missing"""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        is_valid, missing = validate_input_data(df, ['a', 'b', 'c', 'd'])
        assert is_valid == False
        assert 'c' in missing
        assert 'd' in missing


class TestCalculateRevenue:
    """Tests for calculate_revenue function"""
    
    def test_revenue_calculation(self):
        """Test revenue is calculated correctly"""
        price = np.array([10, 20, 30])
        quantity = np.array([5, 3, 2])
        result = calculate_revenue(price, quantity)
        expected = np.array([50, 60, 60])
        np.testing.assert_array_equal(result, expected)
    
    def test_scalar_inputs(self):
        """Test with scalar inputs"""
        result = calculate_revenue(10.0, 5.0)
        assert result == 50.0


class TestCalculateProfit:
    """Tests for calculate_profit function"""
    
    def test_profit_calculation(self):
        """Test profit calculation"""
        revenue = np.array([100, 200, 300])
        cost = np.array([80, 150, 250])
        result = calculate_profit(revenue, cost)
        expected = np.array([20, 50, 50])
        np.testing.assert_array_equal(result, expected)


class TestCalculateMargin:
    """Tests for calculate_margin function"""
    
    def test_margin_calculation(self):
        """Test margin calculation"""
        price = np.array([100, 200])
        cost = np.array([80, 150])
        result = calculate_margin(price, cost)
        expected = np.array([20.0, 25.0])  # (100-80)/100*100, (200-150)/200*100
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_zero_price(self):
        """Test margin with zero price uses default"""
        price = np.array([0])
        cost = np.array([50])
        result = calculate_margin(price, cost)
        assert result[0] == 0.0  # Default when price is zero


class TestCalculateMAPE:
    """Tests for MAPE calculation"""
    
    def test_basic_mape(self):
        """Test basic MAPE calculation"""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 180, 330])
        result = calculate_mape(y_true, y_pred)
        # Expected: mean(|0.1|, |0.1|, |0.1|) * 100 = 10%
        assert result == pytest.approx(10.0)
    
    def test_mape_with_zeros(self):
        """Test MAPE ignores zero actual values"""
        y_true = np.array([100, 0, 200])  # One zero
        y_pred = np.array([110, 50, 180])
        result = calculate_mape(y_true, y_pred)
        # Should only consider non-zero values
        assert result > 0
    
    def test_all_zeros(self):
        """Test MAPE with all zeros returns 0"""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([10, 20, 30])
        result = calculate_mape(y_true, y_pred)
        assert result == 0.0


class TestPriceElasticity:
    """Tests for price elasticity calculation"""
    
    def test_negative_elasticity_normal_goods(self):
        """Test that normal goods have negative elasticity"""
        prices = np.array([10, 15, 20])
        quantities = np.array([100, 70, 50])
        
        elasticity = calculate_price_elasticity(prices, quantities)
        
        # Most values should be negative (normal goods)
        assert elasticity[1] < 0
    
    def test_classify_elastic(self):
        """Test elastic classification"""
        result = classify_elasticity(-1.5)
        assert 'Elastic' in result
    
    def test_classify_inelastic(self):
        """Test inelastic classification"""
        result = classify_elasticity(-0.3)
        assert 'Inelastic' in result


class TestValidationResult:
    """Tests for ValidationResult dataclass"""
    
    def test_bool_true(self):
        """Test ValidationResult returns True when valid"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert bool(result) == True
    
    def test_bool_false(self):
        """Test ValidationResult returns False when invalid"""
        result = ValidationResult(is_valid=False, errors=['Error'], warnings=[])
        assert bool(result) == False


class TestValidatePriceRange:
    """Tests for validate_price_range function"""
    
    def test_valid_range(self):
        """Test valid price range"""
        result = validate_price_range(0.20)
        assert result.is_valid == True
    
    def test_invalid_range_low(self):
        """Test price range too low"""
        result = validate_price_range(0.01)
        assert result.is_valid == False


class TestValidateProductData:
    """Tests for validate_product_data function"""
    
    def test_valid_product(self):
        """Test valid product data"""
        product = pd.Series({
            'unit_price': 100,
            'freight_price': 10,
            'comp_1': 90,
            'comp_2': 110,
            'comp_3': 100
        })
        result = validate_product_data(product)
        assert result.is_valid == True
    
    def test_invalid_negative_price(self):
        """Test negative price fails"""
        product = pd.Series({
            'unit_price': -100,
            'freight_price': 10,
            'comp_1': 90,
            'comp_2': 110,
            'comp_3': 100
        })
        result = validate_product_data(product)
        assert result.is_valid == False


# Integration tests
class TestIntegration:
    """Integration tests for the full pipeline"""
    
    def test_full_feature_engineering_pipeline(self):
        """Test full feature engineering pipeline"""
        df = pd.DataFrame({
            'unit_price': [100, 150],
            'freight_price': [10, 15],
            'comp_1': [95, 145],
            'comp_2': [105, 155],
            'comp_3': [100, 150]
        })
        
        # Apply all transformations
        df = calculate_competitor_features(df)
        df = calculate_cost_features(df)
        
        # Check all expected columns exist
        expected_cols = [
            'diff_comp_1', 'ratio_comp_1',
            'diff_comp_2', 'ratio_comp_2',
            'diff_comp_3', 'ratio_comp_3',
            'total_cost'
        ]
        
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
