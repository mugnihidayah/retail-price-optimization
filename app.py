import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import numpy as np

# CONFIGURATION
st.set_page_config(
    page_title="Retail Pricing Intelligence",
    page_icon="💰",
    layout="wide"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #30333f;
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# LOAD DATA & MODEL
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/train_data.csv')
    return df

@st.cache_resource
def load_model():
    model = joblib.load('models/pricing_model.pkl')
    return model

try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# OPTIMIZATION ENGINE (The Brain)
def optimize_product_price(row_data, model, price_range=0.20):
    current_price = row_data['unit_price']
    
    # Generate 50 simulated price points (+/- 20%)
    test_prices = np.linspace(current_price * (1 - price_range), 
                              current_price * (1 + price_range), 
                              50)
    
    revenues = []
    
    # Take the name of the feature that the model expects
    feature_names = model.feature_names_in_
    
    # Create a simulation dataframe
    sim_df = pd.DataFrame([row_data] * len(test_prices))
    sim_df['unit_price'] = test_prices
    
    # Update derivative features
    sim_df['diff_comp_1'] = sim_df['unit_price'] - sim_df['comp_1']
    sim_df['ratio_comp_1'] = sim_df['unit_price'] / sim_df['comp_1']
    sim_df['diff_comp_2'] = sim_df['unit_price'] - sim_df['comp_2']
    sim_df['ratio_comp_2'] = sim_df['unit_price'] / sim_df['comp_2']
    sim_df['diff_comp_3'] = sim_df['unit_price'] - sim_df['comp_3']
    sim_df['ratio_comp_3'] = sim_df['unit_price'] / sim_df['comp_3']
    sim_df['total_cost'] = sim_df['unit_price'] + sim_df['freight_price']
    
    # Filter columns according to model input
    X_sim = sim_df[feature_names]
    
    # Quantity Prediction
    pred_qtys = model.predict(X_sim)
    pred_qtys = np.maximum(pred_qtys, 0) # Qty tidak boleh minus
    
    # Calculate Revenue
    pred_revenues = test_prices * pred_qtys
    
    # Find Optimal Results
    max_idx = np.argmax(pred_revenues)
    return test_prices, pred_revenues, test_prices[max_idx], pred_revenues[max_idx]

# UI: SIDEBAR
st.sidebar.header("🎛️ Simulation Controls")

# Category Selection
cat_columns = [c for c in df.columns if 'product_category_name_' in c]
clean_cats = [c.replace('product_category_name_', '') for c in cat_columns]
selected_cat_clean = st.sidebar.selectbox("Select Category", clean_cats)

# Filter the dataframe by category
cat_col_name = f'product_category_name_{selected_cat_clean}'
df_filtered = df[df[cat_col_name] == 1].reset_index(drop=True)

if len(df_filtered) == 0:
    st.warning("No products found in this category.")
    st.stop()

# Product Selection
product_index = st.sidebar.selectbox(
    "Select Product (Index)", 
    df_filtered.index,
    format_func=lambda x: f"Product #{x} (Price: ${df_filtered.loc[x, 'unit_price']})"
)

selected_product = df_filtered.loc[product_index]

# UI: MAIN PAGE
st.title("💰 AI Pricing Strategy Dashboard")
st.markdown("### Prescriptive Analytics: From Prediction to Profit")

# 2 Column Layout: Product Details & Competitors
col1, col2 = st.columns(2)

with col1:
    st.info("📦 **Product Details**")
    st.write(f"**Base Price:** ${selected_product['unit_price']:.2f}")
    st.write(f"**Freight Cost:** ${selected_product['freight_price']:.2f}")
    st.write(f"**Product Score:** {selected_product['product_score']}/5.0")

with col2:
    st.warning("⚔️ **Market Competition**")
    st.write(f"**Competitor 1:** ${selected_product['comp_1']:.2f}")
    st.write(f"**Competitor 2:** ${selected_product['comp_2']:.2f}")
    st.write(f"**Competitor 3:** ${selected_product['comp_3']:.2f}")

st.divider()

# ACTION BUTTON
if st.button("🚀 Run AI Optimization", type="primary", use_container_width=True):
    
    with st.spinner("Analyzing Market Elasticity..."):
        prices, revenues, opt_price, max_rev = optimize_product_price(selected_product, model)
        
        current_rev = selected_product['unit_price'] * (model.predict(pd.DataFrame([selected_product])[model.feature_names_in_])[0])
        
        # KPI METRICS
        kpi1, kpi2, kpi3 = st.columns(3)
        
        kpi1.metric(
            label="Current Price",
            value=f"${selected_product['unit_price']:.2f}",
        )
        
        kpi2.metric(
            label="🚀 Optimal Price (AI)",
            value=f"${opt_price:.2f}",
            delta=f"{((opt_price - selected_product['unit_price'])/selected_product['unit_price'])*100:.1f}%",
            delta_color="normal"
        )
        
        kpi3.metric(
            label="Potential Revenue Uplift",
            value=f"${max_rev:.2f}",
            delta=f"+${max_rev - current_rev:.2f}",
            delta_color="inverse"
        )
        
        # INTERACTIVE CHART
        st.subheader("📊 Revenue Optimization Curve")
        
        fig = go.Figure()
        
        # Revenue Line
        fig.add_trace(go.Scatter(
            x=prices, y=revenues,
            mode='lines',
            name='Predicted Revenue',
            line=dict(color='#00CC96', width=3)
        ))
        
        # Current Point (Red)
        fig.add_trace(go.Scatter(
            x=[selected_product['unit_price']], y=[current_rev],
            mode='markers',
            name='Current Price',
            marker=dict(color='#EF553B', size=12, symbol='x')
        ))
        
        # Optimal Point (Blue)
        fig.add_trace(go.Scatter(
            x=[opt_price], y=[max_rev],
            mode='markers',
            name='Optimal Price',
            marker=dict(color='#636EFA', size=15)
        ))
        
        fig.update_layout(
            title="Price vs Predicted Revenue",
            xaxis_title="Unit Price ($)",
            yaxis_title="Total Revenue ($)",
            hovermode="x unified",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # AI STRATEGY
        st.success(f"**Strategic Insight:** Set price to **${opt_price:.2f}**. This maximizes revenue by balancing volume and margin based on competitor prices.")

else:
    st.info("👈 Select a product from the sidebar and click 'Run AI Optimization' to see result.")