"""Beautiful Streamlit frontend for bird classification."""

import os
from io import BytesIO

import httpx
import streamlit as st
import structlog
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from PIL import Image

API_URL = os.getenv("API_URL")
API_TIMEOUT = 30


def _inject_trace_context(
    logger: structlog.types.WrappedLogger,
    method: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Inject active OTel trace/span IDs so Grafana can correlate logs ↔ traces."""
    span = otel_trace.get_current_span()
    if span.is_recording():
        ctx = span.get_span_context()
        event_dict['traceID'] = format(ctx.trace_id, '032x')
        event_dict['spanID'] = format(ctx.span_id, '016x')
    return event_dict


structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        _inject_trace_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt='iso'),
        structlog.processors.JSONRenderer(),
    ],
)

logger = structlog.get_logger()


def setup_telemetry() -> None:
    """Configure OpenTelemetry tracing for frontend service."""
    resource = Resource.create({
        'service.name': 'bird-classification-frontend',
        'service.version': '1.0.0',
    })
    provider = TracerProvider(resource=resource)
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://alloy.aide-monitoring.svc.cluster.local:4317")
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)),
    )
    otel_trace.set_tracer_provider(provider)
    HTTPXClientInstrumentor().instrument()
    logger.info('OpenTelemetry tracing initialized for frontend service.')


setup_telemetry()

# Get global tracer for creating spans
tracer = otel_trace.get_tracer(__name__)

APP_TITLE = "Bird Classification"
APP_ICON = "🐦"

THEME_PRIMARY_COLOR = "#667eea"
THEME_SECONDARY_COLOR = "#764ba2"

# Page configuration
st.set_page_config(
    page_title="🐦 Bird Classification",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for beautiful styling
def inject_custom_css():
    """Inject custom CSS for beautiful styling."""
    css = """
    <style>
    /* Main styling */
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .header-container h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
    }
    
    .header-container p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }
    
    .result-card h2 {
        font-size: 2.5rem;
        margin: 0.5rem 0;
        font-weight: 700;
    }
    
    .result-card p {
        font-size: 1.2rem;
        margin: 0.5rem 0;
        opacity: 0.95;
    }
    
    /* Probability bar */
    .probability-bar {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .probability-fill {
        background: linear-gradient(90deg, #4ade80 0%, #22c55e 100%);
        height: 40px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: white;
        font-size: 1.1rem;
    }
    
    /* Alternative classes */
    .alternative-class {
        background: white;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: 0.2s;
    }
    
    .alternative-class:hover {
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.2);
        transform: translateX(4px);
    }
    
    .alternative-class .class-name {
        font-weight: 600;
        color: #333;
        flex: 1;
    }
    
    .alternative-class .probability {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 700;
        margin-left: 1rem;
    }
    
    /* Upload area */
    .upload-area {
        border: 3px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f0f4ff;
        color: #667eea;
        font-weight: 600;
        cursor: pointer;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: 0.3s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Loading spinner */
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Success message */
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Error message */
    .error-box {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header():
    """Render the beautiful header."""
    st.markdown("""
    <div class="header-container">
        <h1>🐦 Bird Classification</h1>
        <p>Identify bird species using advanced AI technology</p>
    </div>
    """, unsafe_allow_html=True)


def handle_image_upload_and_predict(
    uploaded_file,
    api_url: str,
    timeout: float = 30.0,
):
    """Handle image upload and prediction."""
    if uploaded_file is None:
        return None
    
    # Create root span for this prediction request
    with tracer.start_as_current_span("predict_bird_species") as span:
        span.set_attribute("filename", uploaded_file.name)
        span.set_attribute("content_type", uploaded_file.type)
        
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            logger.info('Image loaded successfully.', extra={'filename': uploaded_file.name})
            
            # Show loading message
            with st.spinner("🔍 Analyzing image and classifying bird species..."):
                # Make prediction request
                logger.info('Sending prediction request to API.')
                with httpx.Client(timeout=timeout) as client:
                    files = {
                        'file': (uploaded_file.name, BytesIO(uploaded_file.getvalue()), uploaded_file.type)
                    }
                    response = client.post(
                        f"{api_url}/predict",
                        files=files,
                    )
                    response.raise_for_status()
                    prediction = response.json()
            
            logger.info('Prediction received successfully.', extra={'predicted_class': prediction.get('predicted_class')})
            span.set_attribute("predicted_class", prediction.get('predicted_class'))
            span.set_attribute("probability", prediction.get('probability', 0))
            return image, prediction
        except httpx.HTTPError as e:
            logger.exception('HTTP error during prediction request.', extra={'error': str(e)})
            span.set_attribute("error", str(e))
            st.error(f"❌ API Error: {str(e)}", icon="🚫")
            return None
        except Exception as e:
            logger.exception('Error processing image.', extra={'error': str(e)})
            span.set_attribute("error", str(e))
            st.error(f"❌ Error processing image: {str(e)}", icon="🚫")
            return None


def render_prediction_results(image: Image.Image, prediction: dict):
    """Render prediction results beautifully."""
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📷 Analyzed Image")
        st.image(image, caption="Uploaded bird image")
    
    with col2:
        st.markdown("### 🎯 Classification Results")
        
        # Main result card
        predicted_class = prediction.get('predicted_class', 'Unknown')
        probability = prediction.get('probability', 0)
        
        st.markdown(f"""
        <div class="result-card">
            <p style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">🏆 TOP PREDICTION</p>
            <h2>{predicted_class}</h2>
            <div class="probability-bar">
                <div class="probability-fill" style="width: {min(probability * 100, 100)}%; transition: width 0.5s ease;">
                    {probability * 100:.1f}% Confidence
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Alternative predictions
        st.markdown("### 🔄 Alternative Species")
        alternatives = prediction.get('top_3_alternatives', [])
        
        if alternatives:
            for idx, alt in enumerate(alternatives, 1):
                class_name = alt.get('class', 'Unknown')
                alt_prob = alt.get('probability', 0)
                st.markdown(f"""
                <div class="alternative-class">
                    <span class="class-name">#{idx} {class_name}</span>
                    <span class="probability">{alt_prob * 100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No alternative predictions available.")


def render_sidebar():
    """Render sidebar with configuration."""
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        This application uses advanced deep learning models to classify bird species from images.
        
        **Features:**
        - 🐦 Supports 50+ bird species
        - 🎯 High accuracy predictions
        - 📊 Confidence scores
        - 🎨 Beautiful & intuitive UI
        
        **How to use:**
        1. Upload a bird image
        2. Wait for analysis
        3. View results instantly
        """)
        
        return API_URL, API_TIMEOUT


def main():
    """Main entry point for Streamlit."""
    logger.info('Rendering bird classification frontend application.')
    inject_custom_css()
    render_header()
    
    # Sidebar
    api_url, timeout = render_sidebar()
    
    # Main content
    st.markdown("---")
    
    # Create two columns for upload options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "label",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help="Upload a JPG or PNG image of a bird",
            label_visibility="collapsed",
        )
    
    with col2:
        st.markdown("### 🎯 Quick Tips")
        st.info("""
        ✅ **Best Results With:**
        - Clear, well-lit images
        - Bird in focus
        - JPG or PNG format
        - 800x600px or larger
        """)
    
    # Process uploads
    if uploaded_file is not None:
        st.markdown("---")
        result = handle_image_upload_and_predict(uploaded_file, api_url, timeout)
        if result:
            image, prediction = result
            render_prediction_results(image, prediction)
            
            # Download info
            st.success(f"✅ Successfully classified as **{prediction.get('predicted_class')}**!")
    
    else:
        st.markdown("---")
        st.markdown("""
        <div class="info-box">
            <strong>👆 Get Started:</strong> Upload a bird image above to begin instant classification!
        </div>
        """, unsafe_allow_html=True)
        

if __name__ == "__main__":
    main()

