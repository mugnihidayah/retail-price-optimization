"""
Enhanced Retail Pricing Intelligence Dashboard
Features: Batch optimization, export, confidence intervals, improved visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import logging

# Import custom modules
try:
    from utils import (
        safe_division, calculate_competitor_features, 
        calculate_cost_features, clip_predictions,
        format_currency, format_percentage,
        get_category_columns, get_category_names,
        validate_price_range, validate_product_data,
        calculate_price_elasticity, classify_elasticity
    )
    from model_training import PricingModel
    from config import app_config, optimization_config
    USE_ENHANCED_MODEL = True
    USE_VALIDATION = True
except ImportError as e:
    logging.warning(f"Could not import enhanced modules: {e}")
    USE_ENHANCED_MODEL = False
    USE_VALIDATION = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Retail Pricing Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border: 1px solid #30333f;
        padding: 20px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA & MODEL LOADING
# ============================================================================
@st.cache_data
def load_data():
    """Load processed training data"""
    try:
        df = pd.read_csv('data/processed/train_data.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model():
    """Load pricing model with fallback"""
    try:
        # Try enhanced model first
        if USE_ENHANCED_MODEL:
            try:
                model = PricingModel.load('models/pricing_model_enhanced.pkl')
                logger.info("Loaded enhanced pricing model successfully")
                return model, True
            except FileNotFoundError:
                logger.warning("Enhanced model not found, trying fallback...")
            except ValueError as e:
                logger.error(f"Enhanced model corrupted: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading enhanced model: {e}")
        
        # Fallback to original model
        model = joblib.load('models/pricing_model.pkl')
        logger.info("Loaded original pricing model")
        return model, False
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first.")
        logger.error("Model file not found at models/pricing_model.pkl")
        return None, False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        logger.exception("Failed to load model")
        return None, False

# Load data and model
df = load_data()
model_result = load_model()

if df is None or model_result[0] is None:
    st.error("Critical Error: Could not load data or model")
    st.stop()

model, is_enhanced = model_result

# ============================================================================
# OPTIMIZATION ENGINE
# ============================================================================
def optimize_product_price(row_data, model, price_range=0.20, num_points=50):
    """
    Optimize price for a single product with input validation.
    
    Args:
        row_data: Product data as Series or dict
        model: Trained pricing model
        price_range: Range of prices to test (+/- percentage)
        num_points: Number of price points to simulate
        
    Returns:
        Tuple of (test_prices, revenues, optimal_price, max_revenue, confidence_info)
        
    Raises:
        ValueError: If input validation fails
    """
    # Input validation
    if USE_VALIDATION:
        price_validation = validate_price_range(price_range)
        if not price_validation.is_valid:
            raise ValueError(f"Invalid price range: {price_validation.errors}")
        if price_validation.warnings:
            for warning in price_validation.warnings:
                logger.warning(warning)
        
        product_validation = validate_product_data(row_data)
        if not product_validation.is_valid:
            raise ValueError(f"Invalid product data: {product_validation.errors}")
        if product_validation.warnings:
            for warning in product_validation.warnings:
                logger.warning(warning)
    
    current_price = row_data['unit_price']
    
    # Sanity check on price
    if current_price <= 0:
        raise ValueError(f"Invalid current price: {current_price}")
    
    # Generate price points
    test_prices = np.linspace(
        current_price * (1 - price_range),
        current_price * (1 + price_range),
        num_points
    )
    
    # Get feature names from model
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    elif hasattr(model, 'feature_names'):
        feature_names = model.feature_names
    else:
        feature_names = [c for c in row_data.index if c != 'qty']
    
    # Create simulation dataframe
    sim_df = pd.DataFrame([row_data] * len(test_prices))
    sim_df['unit_price'] = test_prices
    
    # Update derivative features with safe division
    for i in range(1, 4):
        comp_col = f'comp_{i}'
        if comp_col in sim_df.columns:
            sim_df[f'diff_comp_{i}'] = sim_df['unit_price'] - sim_df[comp_col]
            # Safe division to avoid division by zero
            sim_df[f'ratio_comp_{i}'] = np.where(
                sim_df[comp_col] != 0,
                sim_df['unit_price'] / sim_df[comp_col],
                1.0
            )
    
    # Update total cost
    if 'freight_price' in sim_df.columns:
        sim_df['total_cost'] = sim_df['unit_price'] + sim_df['freight_price']
    
    # Filter columns according to model input
    X_sim = sim_df[feature_names]
    
    # Make predictions
    if hasattr(model, 'predict_with_uncertainty') and is_enhanced:
        pred_qtys, pred_stds = model.predict_with_uncertainty(X_sim)
    else:
        pred_qtys = model.predict(X_sim)
        pred_stds = np.zeros_like(pred_qtys)
    
    # Clip negative predictions
    pred_qtys = np.maximum(pred_qtys, 0)
    
    # Calculate revenues
    pred_revenues = test_prices * pred_qtys
    
    # Calculate confidence bounds
    revenue_lower = test_prices * np.maximum(pred_qtys - 2*pred_stds, 0)
    revenue_upper = test_prices * (pred_qtys + 2*pred_stds)
    
    # Find optimal price
    max_idx = np.argmax(pred_revenues)
    optimal_price = test_prices[max_idx]
    max_revenue = pred_revenues[max_idx]
    
    confidence_info = {
        'pred_stds': pred_stds,
        'revenue_lower': revenue_lower,
        'revenue_upper': revenue_upper,
        'optimal_qty': pred_qtys[max_idx],
        'optimal_std': pred_stds[max_idx] if len(pred_stds) > 0 else 0
    }
    
    return test_prices, pred_revenues, optimal_price, max_revenue, confidence_info

def batch_optimize(df, model, price_range=0.20):
    """
    Optimize prices for all products in dataframe
    
    Returns:
        DataFrame with optimization results
    """
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (_, row) in enumerate(df.iterrows()):
        # Update progress
        progress = (idx + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Optimizing product {idx + 1} of {len(df)}...")
        
        try:
            prices, revenues, opt_price, max_rev, conf = optimize_product_price(
                row, model, price_range
            )
            
            # Calculate current revenue
            current_price = row['unit_price']
            current_df = pd.DataFrame([row])
            if hasattr(model, 'feature_names_in_'):
                current_qty = model.predict(current_df[model.feature_names_in_])[0]
            elif hasattr(model, 'feature_names'):
                current_qty = model.predict(current_df[model.feature_names])[0]
            else:
                current_qty = model.predict(current_df[[c for c in row.index if c != 'qty']])[0]
            
            current_rev = current_price * max(current_qty, 0)
            
            results.append({
                'index': idx,
                'current_price': current_price,
                'optimal_price': opt_price,
                'price_change': opt_price - current_price,
                'price_change_pct': ((opt_price - current_price) / current_price) * 100,
                'current_revenue': current_rev,
                'optimal_revenue': max_rev,
                'revenue_uplift': max_rev - current_rev,
                'revenue_uplift_pct': ((max_rev - current_rev) / current_rev * 100) if current_rev > 0 else 0,
                'action': 'Increase' if opt_price > current_price else ('Decrease' if opt_price < current_price else 'Maintain')
            })
        except Exception as e:
            logger.error(f"Error optimizing product {idx}: {e}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def convert_df_to_excel(df):
    """Convert DataFrame to Excel bytes for download"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Optimization Results')
    return output.getvalue()

def convert_df_to_csv(df):
    """Convert DataFrame to CSV bytes for download"""
    return df.to_csv(index=False).encode('utf-8')

# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_sidebar():
    """Render sidebar with controls"""
    st.sidebar.header("🎛️ Controls")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Optimization Mode",
        ["Single Product", "Batch Optimization"],
        help="Single: Optimize one product at a time. Batch: Optimize all products in category."
    )
    
    # Category selection
    cat_columns = get_category_columns(df) if 'get_category_columns' in dir() else [c for c in df.columns if 'product_category_name_' in c]
    clean_cats = [c.replace('product_category_name_', '') for c in cat_columns]
    selected_cat_clean = st.sidebar.selectbox("📦 Select Category", clean_cats)
    
    # Filter dataframe
    cat_col_name = f'product_category_name_{selected_cat_clean}'
    df_filtered = df[df[cat_col_name] == 1].reset_index(drop=True)
    
    if len(df_filtered) == 0:
        st.sidebar.warning("No products found in this category.")
        return None, None, mode, selected_cat_clean
    
    st.sidebar.info(f"📊 {len(df_filtered)} products in category")
    
    # Product selection (for single mode)
    selected_product = None
    if mode == "Single Product":
        product_index = st.sidebar.selectbox(
            "🏷️ Select Product",
            df_filtered.index,
            format_func=lambda x: f"Product #{x} | ${df_filtered.loc[x, 'unit_price']:.2f} | Score: {df_filtered.loc[x, 'product_score']}"
        )
        selected_product = df_filtered.loc[product_index]
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        price_range = st.slider(
            "Price Range (+/-)",
            min_value=0.05,
            max_value=0.50,
            value=0.20,
            step=0.05,
            help="Range of prices to test around current price"
        )
        st.session_state['price_range'] = price_range
    
    return df_filtered, selected_product, mode, selected_cat_clean

def render_product_details(product):
    """Render product details cards"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📦 Product Details")
        st.markdown(f"""
        <div class="info-card">
            <p><strong>Base Price:</strong> ${product['unit_price']:.2f}</p>
            <p><strong>Freight Cost:</strong> ${product['freight_price']:.2f}</p>
            <p><strong>Product Score:</strong> {product['product_score']}/5.0</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ⚔️ Competition")
        st.markdown(f"""
        <div class="info-card">
            <p><strong>Competitor 1:</strong> ${product['comp_1']:.2f}</p>
            <p><strong>Competitor 2:</strong> ${product['comp_2']:.2f}</p>
            <p><strong>Competitor 3:</strong> ${product['comp_3']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 📈 Market Position")
        avg_comp = (product['comp_1'] + product['comp_2'] + product['comp_3']) / 3
        position = "Above" if product['unit_price'] > avg_comp else "Below"
        diff = abs(product['unit_price'] - avg_comp)
        st.markdown(f"""
        <div class="info-card">
            <p><strong>Avg Competitor Price:</strong> ${avg_comp:.2f}</p>
            <p><strong>Your Position:</strong> {position} by ${diff:.2f}</p>
            <p><strong>Price Ratio:</strong> {product['unit_price']/avg_comp:.2f}x</p>
        </div>
        """, unsafe_allow_html=True)

def render_optimization_results(product, prices, revenues, opt_price, max_rev, conf, current_rev):
    """Render optimization results with visualizations"""
    
    # KPI Metrics
    st.markdown("### 📊 Optimization Results")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric(
            label="Current Price",
            value=f"${product['unit_price']:.2f}"
        )
    
    with kpi2:
        price_change_pct = ((opt_price - product['unit_price']) / product['unit_price']) * 100
        st.metric(
            label="🚀 Optimal Price (AI)",
            value=f"${opt_price:.2f}",
            delta=f"{price_change_pct:+.1f}%"
        )
    
    with kpi3:
        revenue_uplift = max_rev - current_rev
        st.metric(
            label="Potential Revenue",
            value=f"${max_rev:.2f}",
            delta=f"+${revenue_uplift:.2f}"
        )
    
    with kpi4:
        if current_rev > 0:
            uplift_pct = (revenue_uplift / current_rev) * 100
            st.metric(
                label="Revenue Uplift",
                value=f"{uplift_pct:+.1f}%",
                delta=f"${revenue_uplift:.2f}"
            )
        else:
            st.metric(label="Revenue Uplift", value="N/A")
    
    # Confidence interval info
    if conf['optimal_std'] > 0:
        st.info(f"📊 Prediction Confidence: Optimal quantity = {conf['optimal_qty']:.1f} ± {conf['optimal_std']*2:.1f} units (95% CI)")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue curve
        fig = go.Figure()
        
        # Add confidence band if available
        if 'revenue_lower' in conf and conf['revenue_lower'] is not None:
            fig.add_trace(go.Scatter(
                x=np.concatenate([prices, prices[::-1]]),
                y=np.concatenate([conf['revenue_upper'], conf['revenue_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(0, 204, 150, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence'
            ))
        
        # Revenue line
        fig.add_trace(go.Scatter(
            x=prices, y=revenues,
            mode='lines',
            name='Predicted Revenue',
            line=dict(color='#00CC96', width=3)
        ))
        
        # Current point
        fig.add_trace(go.Scatter(
            x=[product['unit_price']], y=[current_rev],
            mode='markers',
            name='Current Price',
            marker=dict(color='#EF553B', size=14, symbol='x')
        ))
        
        # Optimal point
        fig.add_trace(go.Scatter(
            x=[opt_price], y=[max_rev],
            mode='markers',
            name='Optimal Price',
            marker=dict(color='#636EFA', size=16, symbol='star')
        ))
        
        fig.update_layout(
            title="💰 Revenue Optimization Curve",
            xaxis_title="Unit Price ($)",
            yaxis_title="Total Revenue ($)",
            hovermode="x unified",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Competitor comparison
        comp_data = pd.DataFrame({
            'Entity': ['You (Current)', 'You (Optimal)', 'Competitor 1', 'Competitor 2', 'Competitor 3'],
            'Price': [product['unit_price'], opt_price, product['comp_1'], product['comp_2'], product['comp_3']]
        })
        
        colors = ['#EF553B', '#00CC96', '#636EFA', '#AB63FA', '#FFA15A']
        
        fig2 = go.Figure(data=[
            go.Bar(
                x=comp_data['Entity'],
                y=comp_data['Price'],
                marker_color=colors,
                text=[f"${p:.2f}" for p in comp_data['Price']],
                textposition='auto'
            )
        ])
        
        fig2.update_layout(
            title="📊 Price Comparison",
            yaxis_title="Price ($)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Strategic insight
    if opt_price > product['unit_price']:
        action = "📈 **INCREASE PRICE**"
        insight = f"The model suggests raising your price to **${opt_price:.2f}**. This product has inelastic demand - customers won't significantly reduce purchases with this price increase."
    elif opt_price < product['unit_price']:
        action = "📉 **DECREASE PRICE**"
        insight = f"The model suggests lowering your price to **${opt_price:.2f}**. The increased volume will more than compensate for the lower margin."
    else:
        action = "✅ **MAINTAIN PRICE**"
        insight = "Your current price is already optimal. No changes recommended."
    
    st.success(f"{action}: {insight}")

def render_batch_results(results_df, category):
    """Render batch optimization results"""
    st.markdown(f"### 📊 Batch Optimization Results for {category}")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_products = len(results_df)
        st.metric("Total Products", total_products)
    
    with col2:
        increase_count = len(results_df[results_df['action'] == 'Increase'])
        st.metric("Recommend Increase", increase_count)
    
    with col3:
        decrease_count = len(results_df[results_df['action'] == 'Decrease'])
        st.metric("Recommend Decrease", decrease_count)
    
    with col4:
        total_uplift = results_df['revenue_uplift'].sum()
        st.metric("Total Revenue Uplift", f"${total_uplift:,.2f}")
    
    # Results table
    st.markdown("#### 📋 Detailed Results")
    
    display_df = results_df[[
        'index', 'current_price', 'optimal_price', 'price_change_pct',
        'current_revenue', 'optimal_revenue', 'revenue_uplift_pct', 'action'
    ]].copy()
    
    display_df.columns = [
        'Product ID', 'Current Price', 'Optimal Price', 'Price Change %',
        'Current Revenue', 'Optimal Revenue', 'Revenue Uplift %', 'Action'
    ]
    
    # Format columns
    display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}")
    display_df['Optimal Price'] = display_df['Optimal Price'].apply(lambda x: f"${x:.2f}")
    display_df['Price Change %'] = display_df['Price Change %'].apply(lambda x: f"{x:+.1f}%")
    display_df['Current Revenue'] = display_df['Current Revenue'].apply(lambda x: f"${x:.2f}")
    display_df['Optimal Revenue'] = display_df['Optimal Revenue'].apply(lambda x: f"${x:.2f}")
    display_df['Revenue Uplift %'] = display_df['Revenue Uplift %'].apply(lambda x: f"{x:+.1f}%")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            results_df, x='price_change_pct',
            title='Distribution of Recommended Price Changes',
            labels={'price_change_pct': 'Price Change (%)'},
            template='plotly_dark',
            color_discrete_sequence=['#00CC96']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            results_df, names='action',
            title='Recommended Actions',
            template='plotly_dark',
            color_discrete_sequence=['#00CC96', '#EF553B', '#636EFA']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Export buttons
    st.markdown("#### 📥 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = convert_df_to_csv(results_df)
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"price_optimization_{category}.csv",
            mime="text/csv"
        )
    
    with col2:
        excel_data = convert_df_to_excel(results_df)
        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name=f"price_optimization_{category}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.title("💰 AI Pricing Strategy Dashboard")
    st.markdown("### Prescriptive Analytics: From Prediction to Profit")
    
    # Model info
    if is_enhanced:
        st.success("✅ Using Enhanced Model with Cross-Validation & Confidence Intervals")
    else:
        st.info("ℹ️ Using Standard Model. Run `model_training.py` to train enhanced model.")
    
    st.divider()
    
    # Sidebar
    df_filtered, selected_product, mode, category = render_sidebar()
    
    if df_filtered is None:
        st.stop()
    
    price_range = st.session_state.get('price_range', 0.20)
    
    # Main content based on mode
    if mode == "Single Product":
        # Product details
        render_product_details(selected_product)
        st.divider()
        
        # Optimization button
        if st.button("🚀 Run AI Optimization", type="primary", use_container_width=True):
            with st.spinner("Analyzing Market Elasticity..."):
                prices, revenues, opt_price, max_rev, conf = optimize_product_price(
                    selected_product, model, price_range
                )
                
                # Calculate current revenue
                current_df = pd.DataFrame([selected_product])
                if hasattr(model, 'feature_names_in_'):
                    current_qty = model.predict(current_df[model.feature_names_in_])[0]
                elif hasattr(model, 'feature_names'):
                    current_qty = model.predict(current_df[model.feature_names])[0]
                else:
                    feature_cols = [c for c in selected_product.index if c != 'qty']
                    current_qty = model.predict(current_df[feature_cols])[0]
                
                current_rev = selected_product['unit_price'] * max(current_qty, 0)
                
                render_optimization_results(
                    selected_product, prices, revenues, 
                    opt_price, max_rev, conf, current_rev
                )
        else:
            st.info("👈 Select a product and click 'Run AI Optimization' to see results.")
    
    else:  # Batch Optimization
        st.markdown(f"### 🔄 Batch Optimization for {category}")
        st.write(f"This will optimize prices for all {len(df_filtered)} products in the **{category}** category.")
        
        if st.button("🚀 Run Batch Optimization", type="primary", use_container_width=True):
            with st.spinner("Optimizing all products..."):
                results_df = batch_optimize(df_filtered, model, price_range)
            
            if len(results_df) > 0:
                render_batch_results(results_df, category)
            else:
                st.error("No products were successfully optimized.")
        else:
            st.info("👆 Click the button above to start batch optimization.")

if __name__ == "__main__":
    main()
