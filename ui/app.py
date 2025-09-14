# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import tempfile
import sys
sys.path.append('..')
try:
    from core_parser import parse_onnx
    from core_search import run_hybrid_search as run_enhanced_search
    from core_predictor import predict_performance_ai
    from core_rtl import generate_rtl
    MOCK_MODE = False
except ImportError as e:
    st.warning(f"Backend modules not available: {e}. Using mock data.")
    from mock_core_parser import parse_onnx
    from mock_core_search import run_simple_search
    MOCK_MODE = True


def run_optimization():
    """Run the full optimization pipeline."""
    if not st.session_state.get('uploaded_file'):
        st.session_state.error_message = "Please upload an ONNX model to begin."
        st.session_state.run_triggered = False
        return

    st.session_state.error_message = ""
    st.session_state.run_triggered = True
    
    uploaded_file = st.session_state.uploaded_file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        with st.spinner("Analyzing ONNX model architecture..."):
            model_json = parse_onnx(tmp_path)
            st.session_state.model_json = model_json

        constraints = {
            'max_latency_ms': st.session_state.max_latency_ms,
            'max_power_w': st.session_state.max_power_w,
            'max_memory_mb': st.session_state.max_memory_mb
        }
        
        with st.spinner("Searching for optimal hardware configurations..."):
            if MOCK_MODE:
                all_results, top_3_results = run_simple_search(model_json, constraints)
            else:
                # Use enhanced AI-powered search
                top_configs = run_enhanced_search(model_json, constraints)
                all_results = top_configs  # For now, use top configs as all results
                top_3_results = top_configs[:3] if top_configs else []
        
        if not top_3_results:
            st.session_state.error_message = (
                "No hardware configurations found that meet the specified constraints. "
                "Please relax the constraints and try again."
            )
            st.session_state.all_configs = pd.DataFrame(all_results) if all_results else None
            st.session_state.top_configs = None
        else:
            st.session_state.all_configs = pd.DataFrame(all_results)
            st.session_state.top_configs = pd.DataFrame(top_3_results)
            
            # Generate RTL for top configuration
            if not MOCK_MODE and top_3_results:
                with st.spinner("Generating RTL code for best configuration..."):
                    top_config = top_3_results[0]
                    rtl_code = generate_rtl(top_config)
                    st.session_state.rtl_code = rtl_code

    except Exception as e:
        st.session_state.error_message = f"An error occurred during optimization: {e}"
        st.session_state.run_triggered = False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# --- UI CONFIG ---
st.set_page_config(layout="wide")
st.title("🤖AccelFlow: AI-Powered Neural Accelerator Optimizer")
st.markdown(
    "Upload your ONNX model and define your performance constraints. "
    "This tool will explore a range of hardware accelerator designs to find the optimal configuration for your needs."
)

# --- STATE INIT ---
if 'run_triggered' not in st.session_state:
    st.session_state.run_triggered = False
    st.session_state.uploaded_file = None
    st.session_state.max_latency_ms = 50
    st.session_state.max_power_w = 5.0
    st.session_state.model_json = None
    st.session_state.all_configs = None
    st.session_state.top_configs = None
    st.session_state.error_message = ""

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    st.session_state.uploaded_file = st.file_uploader(
        "Upload ONNX Model",
        type=['onnx'],
        help="Upload the .onnx file for the neural network you want to optimize."
    )

    st.session_state.max_latency_ms = st.slider(
        "Max Latency (ms)", 10, 500,
        value=100,  # More realistic default
        step=5, help="Set the maximum acceptable inference time for the model."
    )

    st.session_state.max_power_w = st.slider(
        "Max Power (W)", 1.0, 100.0,  # Increased range for realistic models
        value=50.0,  # More realistic default
        step=1.0, format="%.1fW",
        help="Set the maximum acceptable power consumption for the accelerator."
    )
    
    st.session_state.max_memory_mb = st.slider(
        "Max Memory (MB)", 64, 2048,
        value=512,  # Reasonable default
        step=64, help="Set the maximum acceptable memory usage."
    )

    st.button("Run Optimization", on_click=run_optimization, type="primary", use_container_width=True)

#RTL code display panel
if (st.session_state.run_triggered and st.session_state.top_configs is not None and 
    hasattr(st.session_state, 'rtl_code') and not MOCK_MODE):
    
    st.subheader("RTL Code Generation")
    st.code(st.session_state.rtl_code, language='verilog')
    st.download_button(
        label="Download RTL Code",
        data=st.session_state.rtl_code,
        file_name="systolic_array.v",
        mime="text/plain"
    )

# --- MAIN PANEL ---
# ... existing code ...

# --- MAIN PANEL ---
if st.session_state.error_message:
    st.error(st.session_state.error_message)

if st.session_state.run_triggered and st.session_state.top_configs is not None:
    st.header("Optimization Results")

    # --- Top 3 Recommended ---
    st.subheader("Top 3 Recommended Configurations")
    st.dataframe(st.session_state.top_configs, use_container_width=True, hide_index=True)

    # --- Scatter Plot ---
    st.subheader("Performance Trade-off Analysis")

    if st.session_state.all_configs is not None and not st.session_state.all_configs.empty:
        all_df = st.session_state.all_configs.copy()

        # Build IDs to identify top configs
        top_ids = [tuple(rec.values()) for rec in st.session_state.top_configs.to_dict('records')]
        all_df['id'] = [tuple(rec.values()) for rec in all_df.to_dict('records')]

        # ✅ Add Status column
        all_df['Status'] = all_df['id'].apply(
            lambda x: 'Recommended' if x in top_ids else 'Explored'
        )

        fig = px.scatter(
            all_df,
            x="latency_ms", y="power_w",
            color="Status", symbol="Status",
            hover_data=['array_size', 'precision', 'clock_ghz'],
            labels={
                "latency_ms": "Predicted Latency (ms)",
                "power_w": "Predicted Power (W)",
                "Status": "Configuration Status"
            },
            title="Latency vs. Power for All Explored Configurations",
            color_discrete_map={
                'Recommended': 'rgba(255, 65, 54, 0.9)',  # Red
                'Explored': 'rgba(0, 116, 217, 0.6)'     # Blue
            },
            symbol_map={'Recommended': 'star', 'Explored': 'circle'}
        )

        fig.update_traces(marker=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)

        # --- Full Table with Status ---
        st.subheader("All Explored Configurations")
        st.dataframe(all_df.drop(columns=['id']), use_container_width=True, hide_index=True)

    else:
        st.warning("Could not generate the performance plot as no valid configurations were evaluated.")

elif st.session_state.run_triggered:
    st.info("The optimization process completed, but no configurations met the specified criteria.")

else:
    st.info("Please upload a model and click 'Run Optimization' to see the results.")

