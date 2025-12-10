"""
Utility functions for price optimization
Handles data preprocessing, feature engineering, and validation
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES FOR VALIDATION RESULTS
# ============================================================================
@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self) -> bool:
        return self.is_valid


# ============================================================================
# SAFE MATH OPERATIONS
# ============================================================================
def safe_division(
    numerator: Union[pd.Series, np.ndarray], 
    denominator: Union[pd.Series, np.ndarray], 
    default: float = 1.0
) -> np.ndarray:
    """
    Safely divide two series/arrays, handling division by zero.
    
    Args:
        numerator: Values to divide
        denominator: Values to divide by
        default: Value to use when denominator is zero
        
    Returns:
        Array with division results, using default for zero denominators
    """
    return np.where(denominator != 0, numerator / denominator, default)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def calculate_competitor_features(
    df: pd.DataFrame, 
    price_col: str = 'unit_price'
) -> pd.DataFrame:
    """
    Calculate competitor-related features with safe division.
    
    Args:
        df: DataFrame containing price data
        price_col: Name of the price column
        
    Returns:
        DataFrame with calculated competitor features (diff_comp_*, ratio_comp_*)
    """
    result = df.copy()
    
    for i in range(1, 4):  # comp_1, comp_2, comp_3
        comp_col = f'comp_{i}'
        if comp_col in result.columns:
            # Difference from competitor
            result[f'diff_comp_{i}'] = result[price_col] - result[comp_col]
            
            # Safe ratio calculation
            result[f'ratio_comp_{i}'] = safe_division(
                result[price_col], 
                result[comp_col], 
                default=1.0
            )
    
    return result


def calculate_cost_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cost-related features.
    
    Args:
        df: DataFrame with unit_price and freight_price columns
        
    Returns:
        DataFrame with total_cost column added
    """
    result = df.copy()
    
    if 'unit_price' in result.columns and 'freight_price' in result.columns:
        result['total_cost'] = result['unit_price'] + result['freight_price']
    
    return result


def calculate_price_elasticity(
    prices: np.ndarray,
    quantities: np.ndarray,
    method: str = 'point'
) -> np.ndarray:
    """
    Calculate price elasticity of demand.
    
    Elasticity = (% change in quantity) / (% change in price)
    E = (ΔQ/Q) / (ΔP/P) = (ΔQ/ΔP) * (P/Q)
    
    Args:
        prices: Array of prices
        quantities: Array of corresponding quantities
        method: 'point' for point elasticity, 'arc' for arc elasticity
        
    Returns:
        Array of elasticity values (negative values indicate normal goods)
    """
    if len(prices) < 2:
        return np.array([0.0])
    
    if method == 'point':
        # Point elasticity using finite differences
        delta_q = np.diff(quantities)
        delta_p = np.diff(prices)
        
        # Use midpoint values
        avg_q = (quantities[:-1] + quantities[1:]) / 2
        avg_p = (prices[:-1] + prices[1:]) / 2
        
        # Calculate elasticity with safe division
        elasticity = safe_division(delta_q, delta_p, default=0.0) * safe_division(avg_p, avg_q, default=0.0)
        
        # Pad to match input length
        elasticity = np.concatenate([[elasticity[0]], elasticity])
        
    elif method == 'arc':
        # Arc elasticity using midpoint formula
        elasticity = np.zeros(len(prices))
        for i in range(1, len(prices)):
            q1, q2 = quantities[i-1], quantities[i]
            p1, p2 = prices[i-1], prices[i]
            
            pct_q = (q2 - q1) / ((q1 + q2) / 2) if (q1 + q2) != 0 else 0
            pct_p = (p2 - p1) / ((p1 + p2) / 2) if (p1 + p2) != 0 else 0
            
            elasticity[i] = pct_q / pct_p if pct_p != 0 else 0
        
        elasticity[0] = elasticity[1] if len(elasticity) > 1 else 0
    else:
        raise ValueError(f"Unknown elasticity method: {method}")
    
    return elasticity


def classify_elasticity(elasticity: float) -> str:
    """
    Classify demand elasticity.
    
    Args:
        elasticity: Elasticity coefficient (typically negative)
        
    Returns:
        Classification string
    """
    abs_e = abs(elasticity)
    if abs_e > 1:
        return "Elastic (price sensitive)"
    elif abs_e < 1:
        return "Inelastic (price insensitive)"
    else:
        return "Unit elastic"


# ============================================================================
# INPUT VALIDATION
# ============================================================================
def validate_input_data(
    df: pd.DataFrame, 
    required_cols: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate that required columns exist in the dataframe.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        
    Returns:
        Tuple of (is_valid, list of missing columns)
    """
    missing = [col for col in required_cols if col not in df.columns]
    return len(missing) == 0, missing


def validate_price_range(
    price_range: float,
    min_range: float = 0.05,
    max_range: float = 0.50
) -> ValidationResult:
    """
    Validate price range parameter.
    
    Args:
        price_range: Price range value (as decimal, e.g., 0.20 for 20%)
        min_range: Minimum allowed range
        max_range: Maximum allowed range
        
    Returns:
        ValidationResult with is_valid, errors, and warnings
    """
    errors = []
    warnings = []
    
    if not isinstance(price_range, (int, float)):
        errors.append(f"Price range must be a number, got {type(price_range)}")
    elif price_range < min_range:
        errors.append(f"Price range {price_range:.2%} is below minimum {min_range:.2%}")
    elif price_range > max_range:
        errors.append(f"Price range {price_range:.2%} exceeds maximum {max_range:.2%}")
    
    if price_range > 0.30:
        warnings.append(f"Large price range ({price_range:.2%}) may produce unrealistic results")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def validate_product_data(
    product_data: Union[pd.Series, Dict[str, Any]],
    required_fields: Optional[List[str]] = None
) -> ValidationResult:
    """
    Validate product data for optimization.
    
    Args:
        product_data: Product data as Series or dict
        required_fields: List of required field names
        
    Returns:
        ValidationResult with validation status
    """
    if required_fields is None:
        required_fields = ['unit_price', 'freight_price', 'comp_1', 'comp_2', 'comp_3']
    
    errors = []
    warnings = []
    
    # Check required fields
    if isinstance(product_data, pd.Series):
        data_keys = product_data.index.tolist()
    else:
        data_keys = list(product_data.keys())
    
    for field in required_fields:
        if field not in data_keys:
            errors.append(f"Missing required field: {field}")
    
    # Validate price values
    if 'unit_price' in data_keys:
        price = product_data['unit_price']
        if price <= 0:
            errors.append(f"Invalid unit_price: {price} (must be positive)")
        elif price < 1:
            warnings.append(f"Very low unit_price: {price}")
    
    # Validate competitor prices
    for i in range(1, 4):
        comp_col = f'comp_{i}'
        if comp_col in data_keys:
            comp_price = product_data[comp_col]
            if comp_price < 0:
                errors.append(f"Invalid {comp_col}: {comp_price} (cannot be negative)")
            elif comp_price == 0:
                warnings.append(f"{comp_col} is zero, competitor comparison will use default ratio")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def validate_optimization_params(
    num_points: int = 50,
    min_points: int = 10,
    max_points: int = 200
) -> ValidationResult:
    """
    Validate optimization parameters.
    
    Args:
        num_points: Number of price points to simulate
        min_points: Minimum allowed points
        max_points: Maximum allowed points
        
    Returns:
        ValidationResult
    """
    errors = []
    warnings = []
    
    if not isinstance(num_points, int):
        errors.append(f"num_points must be integer, got {type(num_points)}")
    elif num_points < min_points:
        errors.append(f"num_points ({num_points}) below minimum ({min_points})")
    elif num_points > max_points:
        errors.append(f"num_points ({num_points}) exceeds maximum ({max_points})")
    
    if num_points > 100:
        warnings.append(f"High num_points ({num_points}) may slow optimization")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


# ============================================================================
# PREDICTION UTILITIES
# ============================================================================
def clip_predictions(
    predictions: np.ndarray, 
    min_val: float = 0,
    max_val: Optional[float] = None
) -> np.ndarray:
    """
    Clip predictions to valid range.
    
    Args:
        predictions: Array of predictions
        min_val: Minimum allowed value
        max_val: Maximum allowed value (optional)
        
    Returns:
        Clipped predictions array
    """
    result = np.maximum(predictions, min_val)
    if max_val is not None:
        result = np.minimum(result, max_val)
    return result


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================
def format_currency(value: float, currency: str = "$") -> str:
    """Format a value as currency."""
    return f"{currency}{value:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a value as percentage."""
    return f"{value:,.{decimals}f}%"


def format_change(value: float, decimals: int = 1) -> str:
    """Format a value as a change with +/- sign."""
    return f"{value:+,.{decimals}f}%"


# ============================================================================
# CATEGORY UTILITIES  
# ============================================================================
def get_category_columns(
    df: pd.DataFrame, 
    prefix: str = 'product_category_name_'
) -> List[str]:
    """Get all category columns from dataframe."""
    return [c for c in df.columns if c.startswith(prefix)]


def get_category_names(
    df: pd.DataFrame, 
    prefix: str = 'product_category_name_'
) -> List[str]:
    """Get clean category names from dataframe."""
    cat_cols = get_category_columns(df, prefix)
    return [c.replace(prefix, '') for c in cat_cols]


# ============================================================================
# REVENUE & PROFIT CALCULATIONS
# ============================================================================
def calculate_revenue(
    price: Union[np.ndarray, float], 
    quantity: Union[np.ndarray, float]
) -> Union[np.ndarray, float]:
    """Calculate revenue from price and quantity."""
    return price * quantity


def calculate_profit(
    revenue: Union[np.ndarray, float], 
    cost: Union[np.ndarray, float]
) -> Union[np.ndarray, float]:
    """Calculate profit from revenue and cost."""
    return revenue - cost


def calculate_margin(
    price: Union[np.ndarray, float],
    cost: Union[np.ndarray, float]
) -> Union[np.ndarray, float]:
    """Calculate profit margin as percentage."""
    return safe_division(price - cost, price, default=0.0) * 100


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================
def calculate_mape(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        MAPE as a percentage
    """
    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        return 0.0
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for values.
    
    Args:
        values: Array of values
        confidence: Confidence level (default 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    try:
        from scipy import stats
        n = len(values)
        mean = np.mean(values)
        se = stats.sem(values)
        
        # t-value for confidence interval
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        
        margin = t_val * se
        return (mean - margin, mean + margin)
    except ImportError:
        # Fallback if scipy not available
        n = len(values)
        mean = np.mean(values)
        std = np.std(values)
        se = std / np.sqrt(n)
        # Use z-value approximation (1.96 for 95%)
        z_val = 1.96 if confidence == 0.95 else 1.645
        margin = z_val * se
        return (mean - margin, mean + margin)
