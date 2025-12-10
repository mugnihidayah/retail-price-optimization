"""
Unit Tests for App Optimization Functions
Tests price optimization logic and validation
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    validate_price_range, validate_product_data, validate_optimization_params,
    calculate_price_elasticity, classify_elasticity,
    ValidationResult
)


class TestValidatePriceRange:
    """Tests for validate_price_range function"""
    
    def test_valid_price_range(self):
        """Test valid price range passes validation"""
        result = validate_price_range(0.20)
        assert result.is_valid == True
        assert len(result.errors) == 0
    
    def test_price_range_too_low(self):
        """Test price range below minimum fails"""
        result = validate_price_range(0.01)
        assert result.is_valid == False
        assert len(result.errors) > 0
        assert 'below minimum' in result.errors[0]
    
    def test_price_range_too_high(self):
        """Test price range above maximum fails"""
        result = validate_price_range(0.60)
        assert result.is_valid == False
        assert len(result.errors) > 0
        assert 'exceeds maximum' in result.errors[0]
    
    def test_price_range_warning(self):
        """Test large price range triggers warning"""
        result = validate_price_range(0.35)
        assert result.is_valid == True
        assert len(result.warnings) > 0
        assert 'unrealistic' in result.warnings[0].lower()
    
    def test_validation_result_bool(self):
        """Test ValidationResult can be used as boolean"""
        valid_result = validate_price_range(0.20)
        invalid_result = validate_price_range(0.01)
        
        assert bool(valid_result) == True
        assert bool(invalid_result) == False


class TestValidateProductData:
    """Tests for validate_product_data function"""
    
    def test_valid_product_data(self):
        """Test valid product data passes"""
        product = pd.Series({
            'unit_price': 100.0,
            'freight_price': 10.0,
            'comp_1': 95.0,
            'comp_2': 105.0,
            'comp_3': 100.0
        })
        
        result = validate_product_data(product)
        assert result.is_valid == True
    
    def test_missing_required_field(self):
        """Test missing required field fails"""
        product = pd.Series({
            'unit_price': 100.0,
            'freight_price': 10.0,
            # Missing comp_1, comp_2, comp_3
        })
        
        result = validate_product_data(product)
        assert result.is_valid == False
        assert any('Missing required field' in e for e in result.errors)
    
    def test_negative_price(self):
        """Test negative price fails validation"""
        product = pd.Series({
            'unit_price': -10.0,
            'freight_price': 10.0,
            'comp_1': 95.0,
            'comp_2': 105.0,
            'comp_3': 100.0
        })
        
        result = validate_product_data(product)
        assert result.is_valid == False
        assert any('must be positive' in e for e in result.errors)
    
    def test_zero_competitor_warning(self):
        """Test zero competitor price triggers warning"""
        product = pd.Series({
            'unit_price': 100.0,
            'freight_price': 10.0,
            'comp_1': 0.0,  # Zero competitor
            'comp_2': 105.0,
            'comp_3': 100.0
        })
        
        result = validate_product_data(product)
        assert result.is_valid == True  # Still valid
        assert len(result.warnings) > 0
        assert any('zero' in w.lower() for w in result.warnings)
    
    def test_dict_input(self):
        """Test that dict input works same as Series"""
        product = {
            'unit_price': 100.0,
            'freight_price': 10.0,
            'comp_1': 95.0,
            'comp_2': 105.0,
            'comp_3': 100.0
        }
        
        result = validate_product_data(product)
        assert result.is_valid == True


class TestValidateOptimizationParams:
    """Tests for validate_optimization_params function"""
    
    def test_valid_params(self):
        """Test valid parameters pass"""
        result = validate_optimization_params(num_points=50)
        assert result.is_valid == True
    
    def test_too_few_points(self):
        """Test too few points fails"""
        result = validate_optimization_params(num_points=5)
        assert result.is_valid == False
    
    def test_too_many_points(self):
        """Test too many points fails"""
        result = validate_optimization_params(num_points=500)
        assert result.is_valid == False
    
    def test_high_points_warning(self):
        """Test high number of points triggers warning"""
        result = validate_optimization_params(num_points=150)
        assert result.is_valid == True
        assert len(result.warnings) > 0


class TestPriceElasticity:
    """Tests for price elasticity calculation"""
    
    def test_basic_elasticity(self):
        """Test basic elasticity calculation"""
        prices = np.array([10, 11, 12, 13, 14])
        quantities = np.array([100, 95, 90, 85, 80])
        
        elasticity = calculate_price_elasticity(prices, quantities)
        
        assert len(elasticity) == len(prices)
        # Negative elasticity for normal goods (price up, quantity down)
        assert all(e <= 0 for e in elasticity[1:])
    
    def test_single_point(self):
        """Test elasticity with single point"""
        prices = np.array([10])
        quantities = np.array([100])
        
        elasticity = calculate_price_elasticity(prices, quantities)
        
        assert len(elasticity) == 1
        assert elasticity[0] == 0.0
    
    def test_arc_elasticity(self):
        """Test arc elasticity method"""
        prices = np.array([10, 12])
        quantities = np.array([100, 90])
        
        elasticity = calculate_price_elasticity(prices, quantities, method='arc')
        
        assert len(elasticity) == 2
    
    def test_invalid_method(self):
        """Test invalid method raises error"""
        prices = np.array([10, 12])
        quantities = np.array([100, 90])
        
        with pytest.raises(ValueError, match="Unknown elasticity method"):
            calculate_price_elasticity(prices, quantities, method='invalid')


class TestClassifyElasticity:
    """Tests for elasticity classification"""
    
    def test_elastic(self):
        """Test elastic classification"""
        result = classify_elasticity(-2.0)
        assert 'Elastic' in result
        assert 'sensitive' in result.lower()
    
    def test_inelastic(self):
        """Test inelastic classification"""
        result = classify_elasticity(-0.5)
        assert 'Inelastic' in result
        assert 'insensitive' in result.lower()
    
    def test_unit_elastic(self):
        """Test unit elastic classification"""
        result = classify_elasticity(-1.0)
        assert 'Unit elastic' in result
    
    def test_positive_elasticity(self):
        """Test positive elasticity (Giffen/Veblen goods)"""
        result = classify_elasticity(1.5)
        assert 'Elastic' in result


class TestValidationResultDataclass:
    """Tests for ValidationResult dataclass"""
    
    def test_creation(self):
        """Test creating ValidationResult"""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=['Some warning']
        )
        
        assert result.is_valid == True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
    
    def test_bool_conversion(self):
        """Test bool conversion based on is_valid"""
        valid = ValidationResult(is_valid=True, errors=[], warnings=[])
        invalid = ValidationResult(is_valid=False, errors=['Error'], warnings=[])
        
        assert bool(valid) == True
        assert bool(invalid) == False
        
        # Test in if statement
        if valid:
            passed = True
        else:
            passed = False
        assert passed == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
