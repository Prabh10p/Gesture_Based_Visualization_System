import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import speech_recognition as sr
from io import BytesIO
import time

# Page configuration
st.set_page_config(
    page_title="Multimodal Data Visualization",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'current_chart' not in st.session_state:
    st.session_state.current_chart = 'bar'
if 'filter_applied' not in st.session_state:
    st.session_state.filter_applied = 'all'
if 'voice_command' not in st.session_state:
    st.session_state.voice_command = ''
if 'gesture_detected' not in st.session_state:
    st.session_state.gesture_detected = ''
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False
if 'selected_charts' not in st.session_state:
    st.session_state.selected_charts = []
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'sample'
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load sample data
@st.cache_data
def load_sample_data():
    """Generate sample sales data"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=12, freq='M')  # Changed from 'ME' to 'M'
    
    data = pd.DataFrame({
        'Month': dates.strftime('%B'),
        'Sales': np.random.randint(1000, 5000, 12),
        'Revenue': np.random.randint(2000, 8000, 12),
        'Profit': np.random.randint(500, 3000, 12),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 12),
        'Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 12)
    })
    return data

def load_uploaded_data(uploaded_file):
    """Load data from uploaded CSV"""
    try:
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def detect_gesture(frame, hands):
    """Detect hand gestures using MediaPipe - IMPROVED VERSION"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    gesture = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            
            # Get finger positions
            landmarks = hand_landmarks.landmark
            
            # Get key points for finger detection
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            ring_tip = landmarks[16]
            ring_pip = landmarks[14]
            pinky_tip = landmarks[20]
            pinky_pip = landmarks[18]
            
            # Count extended fingers (improved logic)
            fingers_up = 0
            
            # Thumb (check horizontal distance)
            if thumb_tip.x < thumb_ip.x:  # For right hand
                fingers_up += 1
            
            # Other fingers (check if tip is above PIP joint)
            if index_tip.y < index_pip.y:
                fingers_up += 1
            if middle_tip.y < middle_pip.y:
                fingers_up += 1
            if ring_tip.y < ring_pip.y:
                fingers_up += 1
            if pinky_tip.y < pinky_pip.y:
                fingers_up += 1
            
            # Detect gestures based on finger count
            if fingers_up == 1:
                gesture = "next_chart"
            elif fingers_up == 2:
                gesture = "previous_chart"
            elif fingers_up == 3:
                gesture = "apply_filter"
            elif fingers_up == 4:
                gesture = "remove_filter"
            elif fingers_up == 0:
                gesture = "toggle_comparison"
            elif fingers_up == 5:
                gesture = "show_all"
            
            # Display finger count on frame
            cv2.putText(frame, f'Fingers: {fingers_up}', (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if gesture:
                cv2.putText(frame, f'Gesture: {gesture}', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return frame, gesture

def process_voice_command():
    """Process voice commands - IMPROVED VERSION"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak clearly!")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            # Listen with longer timeout
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
            
        # Try to recognize
        try:
            command = recognizer.recognize_google(audio).lower()
            return command
        except sr.UnknownValueError:
            return "not_understood"
            
    except sr.WaitTimeoutError:
        return "timeout"
    except sr.RequestError as e:
        return f"error_api: {str(e)}"
    except Exception as e:
        return f"error: {str(e)}"

def execute_voice_command(command):
    """Execute the recognized voice command - IMPROVED"""
    command = command.lower()
    
    # Chart type commands
    if any(word in command for word in ['bar', 'bar chart']):
        st.session_state.current_chart = 'bar'
        return "✅ Switched to Bar Chart"
    elif any(word in command for word in ['line', 'line chart', 'trend']):
        st.session_state.current_chart = 'line'
        return "✅ Switched to Line Chart"
    elif any(word in command for word in ['pie', 'pie chart', 'circle']):
        st.session_state.current_chart = 'pie'
        return "✅ Switched to Pie Chart"
    elif any(word in command for word in ['scatter', 'scatter plot', 'dots']):
        st.session_state.current_chart = 'scatter'
        return "✅ Switched to Scatter Plot"
    elif any(word in command for word in ['heatmap', 'heat map', 'correlation']):
        st.session_state.current_chart = 'heatmap'
        return "✅ Switched to Heatmap"
    
    # Filter commands
    elif any(word in command for word in ['high', 'filter high', 'high values']):
        st.session_state.filter_applied = 'high'
        return "✅ Filtered for high values"
    elif any(word in command for word in ['low', 'filter low', 'low values']):
        st.session_state.filter_applied = 'low'
        return "✅ Filtered for low values"
    elif any(word in command for word in ['remove filter', 'show all', 'clear filter', 'reset']):
        st.session_state.filter_applied = 'all'
        return "✅ Filter removed"
    
    # Comparison mode
    elif any(word in command for word in ['compare', 'comparison']):
        st.session_state.comparison_mode = not st.session_state.comparison_mode
        return f"✅ Comparison mode {'enabled' if st.session_state.comparison_mode else 'disabled'}"
    
    else:
        return f"❌ Command '{command}' not recognized. Try: 'show bar chart', 'filter high', 'compare'"

def apply_filter(data, filter_type):
    """Apply filter to the dataset"""
    if len(data) == 0:
        return data
    
    # Find numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return data
    
    # Use first numeric column for filtering
    filter_col = numeric_cols[0]
    
    if filter_type == 'high':
        return data[data[filter_col] > data[filter_col].median()]
    elif filter_type == 'low':
        return data[data[filter_col] <= data[filter_col].median()]
    else:
        return data

def create_bar_chart(data):
    """Create bar chart using Plotly"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for visualization")
        return None
    
    x_col = data.columns[0]
    y_cols = numeric_cols[:3]  # Use up to 3 numeric columns
    
    fig = px.bar(
        data, 
        x=x_col, 
        y=y_cols,
        title=f'Bar Chart: {", ".join(y_cols)}',
        barmode='group',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
    )
    fig.update_layout(height=500, hovermode='x unified')
    return fig

def create_line_chart(data):
    """Create line chart using Plotly"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for visualization")
        return None
    
    x_col = data.columns[0]
    y_cols = numeric_cols[:3]
    
    fig = px.line(
        data, 
        x=x_col, 
        y=y_cols,
        title=f'Line Chart: {", ".join(y_cols)}',
        markers=True
    )
    fig.update_layout(height=500, hovermode='x unified')
    return fig

def create_pie_chart(data):
    """Create pie chart using Plotly"""
    # Find categorical and numeric columns
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(cat_cols) == 0 or len(num_cols) == 0:
        st.warning("Need both categorical and numeric columns for pie chart")
        return None
    
    grouped_data = data.groupby(cat_cols[0])[num_cols[0]].sum().reset_index()
    
    fig = px.pie(
        grouped_data, 
        values=num_cols[0], 
        names=cat_cols[0],
        title=f'{num_cols[0]} Distribution by {cat_cols[0]}',
        hole=0.3
    )
    fig.update_layout(height=500)
    return fig

def create_scatter_plot(data):
    """Create scatter plot using Plotly"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for scatter plot")
        return None
    
    fig = px.scatter(
        data, 
        x=numeric_cols[0], 
        y=numeric_cols[1],
        size=numeric_cols[2] if len(numeric_cols) > 2 else None,
        title=f'{numeric_cols[0]} vs {numeric_cols[1]}',
        hover_data=data.columns.tolist()
    )
    fig.update_layout(height=500)
    return fig

def create_heatmap(data):
    """Create heatmap using Seaborn"""
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.shape[1] < 2:
        st.warning("Need at least 2 numeric columns for heatmap")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    correlation_data = numeric_data.corr()
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Heatmap')
    return fig

# Main App
def main():
    # Title and description
    st.title("📊 Multimodal Data Visualization System")
    st.markdown("**Control visualizations using gestures, voice commands, or upload your own data**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # DATA SOURCE SELECTION - NEW FEATURE
        st.subheader("📁 Data Source")
        data_source = st.radio(
            "Choose data source:",
            ['Sample Data', 'Upload CSV'],
            key='data_source_radio'
        )
        
        # File uploader
        if data_source == 'Upload CSV':
            uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
            if uploaded_file is not None:
                st.session_state.uploaded_data = load_uploaded_data(uploaded_file)
                if st.session_state.uploaded_data is not None:
                    st.success(f"✅ Loaded {len(st.session_state.uploaded_data)} rows")
                    st.session_state.data_source = 'uploaded'
            else:
                st.info("👆 Upload a CSV file to get started")
        else:
            st.session_state.data_source = 'sample'
        
        st.markdown("---")
        
        # Manual chart selection
        st.subheader("📊 Manual Selection")
        chart_type = st.selectbox(
            "Select Chart Type",
            ['bar', 'line', 'pie', 'scatter', 'heatmap'],
            key='manual_chart'
        )
        if chart_type != st.session_state.current_chart:
            st.session_state.current_chart = chart_type
        
        # Manual filter
        filter_option = st.radio(
            "Data Filter",
            ['all', 'high', 'low'],
            key='manual_filter'
        )
        if filter_option != st.session_state.filter_applied:
            st.session_state.filter_applied = filter_option
        
        st.markdown("---")
        
        # Voice Control - IMPROVED UI
        st.subheader("🎤 Voice Control")
        with st.expander("View Voice Commands", expanded=False):
            st.markdown("""
            **Chart Commands:**
            - "Show bar chart"
            - "Show line chart"
            - "Show pie chart"
            - "Show scatter plot"
            - "Show heatmap"
            
            **Filter Commands:**
            - "Filter high values"
            - "Filter low values"
            - "Remove filter"
            
            **Other:**
            - "Compare charts"
            """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎙️ Voice Command", use_container_width=True):
                with st.spinner("Listening..."):
                    command = process_voice_command()
                    if command not in ['timeout', 'not_understood'] and not command.startswith('error'):
                        result = execute_voice_command(command)
                        st.session_state.voice_command = f"'{command}' → {result}"
                    elif command == 'timeout':
                        st.session_state.voice_command = "⏱️ Timeout - No speech detected"
                    elif command == 'not_understood':
                        st.session_state.voice_command = "❌ Could not understand. Speak clearly and try again."
                    else:
                        st.session_state.voice_command = f"❌ Error: {command}"
        
        with col2:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.voice_command = ''
        
        if st.session_state.voice_command:
            st.info(st.session_state.voice_command)
        
        st.markdown("---")
        
        # Gesture Control
        st.subheader("✋ Gesture Control")
        with st.expander("View Gestures", expanded=False):
            st.markdown("""
            **Hand Gestures:**
            - 👆 1 finger: Next chart
            - ✌️ 2 fingers: Previous chart
            - 🤟 3 fingers: Apply filter
            - 🖖 4 fingers: Remove filter
            - ✊ Fist (0 fingers): Toggle comparison
            - 🖐️ 5 fingers: Show all data
            """)
        
        gesture_enabled = st.checkbox("Enable Gesture Recognition")
        
        st.markdown("---")
        
        # Comparison mode
        st.subheader("📊 Comparison Mode")
        comparison = st.checkbox(
            "Enable Comparison", 
            value=st.session_state.comparison_mode
        )
        st.session_state.comparison_mode = comparison
        
        if comparison:
            selected = st.multiselect(
                "Select charts to compare",
                ['bar', 'line', 'pie', 'scatter'],
                default=st.session_state.selected_charts,
                key='chart_comparison'
            )
            st.session_state.selected_charts = selected
    
    # Load appropriate data
    if st.session_state.data_source == 'uploaded' and st.session_state.uploaded_data is not None:
        data = st.session_state.uploaded_data
    else:
        data = load_sample_data()
    
    filtered_data = apply_filter(data, st.session_state.filter_applied)
    
    # Main content area
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Filtered Records", len(filtered_data))
    with col3:
        st.metric("Filter", st.session_state.filter_applied.upper())
    
    if st.session_state.filter_applied != 'all':
        st.info(f"🔍 Showing {st.session_state.filter_applied.upper()} values only")
    
    st.markdown("---")
    
    # Gesture recognition display
    if gesture_enabled:
        st.subheader("📹 Gesture Recognition")
        
        try:
            # Create MediaPipe hands instance
            hands_detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            FRAME_WINDOW = st.empty()
            gesture_status = st.empty()
            
            camera = cv2.VideoCapture(0)
            
            if not camera.isOpened():
                st.error("❌ Cannot access camera. Please check camera permissions.")
            else:
                stop_button = st.button("⏹️ Stop Gesture Recognition")
                
                last_gesture_time = 0
                gesture_cooldown = 2  # seconds between gesture detections
                
                while not stop_button:
                    ret, frame = camera.read()
                    if not ret:
                        st.error("Cannot read from camera")
                        break
                    
                    frame = cv2.flip(frame, 1)  # Mirror the frame
                    frame, gesture = detect_gesture(frame, hands_detector)
                    
                    current_time = time.time()
                    
                    if gesture and (current_time - last_gesture_time) > gesture_cooldown:
                        st.session_state.gesture_detected = gesture
                        last_gesture_time = current_time
                        
                        # Execute gesture command
                        charts = ['bar', 'line', 'pie', 'scatter', 'heatmap']
                        
                        if gesture == "next_chart":
                            idx = charts.index(st.session_state.current_chart)
                            st.session_state.current_chart = charts[(idx + 1) % len(charts)]
                            gesture_status.success(f"✋ Next Chart: {st.session_state.current_chart.upper()}")
                        elif gesture == "previous_chart":
                            idx = charts.index(st.session_state.current_chart)
                            st.session_state.current_chart = charts[(idx - 1) % len(charts)]
                            gesture_status.success(f"✋ Previous Chart: {st.session_state.current_chart.upper()}")
                        elif gesture == "apply_filter":
                            st.session_state.filter_applied = 'high'
                            gesture_status.success("✋ Filter Applied: HIGH")
                        elif gesture == "remove_filter":
                            st.session_state.filter_applied = 'all'
                            gesture_status.success("✋ Filter Removed")
                        elif gesture == "toggle_comparison":
                            st.session_state.comparison_mode = not st.session_state.comparison_mode
                            gesture_status.success(f"✋ Comparison: {'ON' if st.session_state.comparison_mode else 'OFF'}")
                        elif gesture == "show_all":
                            st.session_state.filter_applied = 'all'
                            gesture_status.success("✋ Showing All Data")
                        
                        st.rerun()
                    
                    FRAME_WINDOW.image(frame, channels="BGR", use_column_width=True)
                
                camera.release()
                hands_detector.close()
                
        except Exception as e:
            st.error(f"Gesture recognition error: {str(e)}")
            st.info("Make sure your camera is connected and permissions are granted.")
    
    st.markdown("---")
    
    # Display charts
    if st.session_state.comparison_mode and st.session_state.selected_charts:
        st.subheader("📊 Chart Comparison Mode")
        
        num_charts = len(st.session_state.selected_charts)
        cols = st.columns(num_charts)
        
        for idx, chart in enumerate(st.session_state.selected_charts):
            with cols[idx]:
                st.markdown(f"**{chart.upper()}**")
                chart_fig = None
                
                if chart == 'bar':
                    chart_fig = create_bar_chart(filtered_data)
                elif chart == 'line':
                    chart_fig = create_line_chart(filtered_data)
                elif chart == 'pie':
                    chart_fig = create_pie_chart(filtered_data)
                elif chart == 'scatter':
                    chart_fig = create_scatter_plot(filtered_data)
                
                if chart_fig:
                    if isinstance(chart_fig, plt.Figure):
                        st.pyplot(chart_fig)
                    else:
                        st.plotly_chart(chart_fig, use_container_width=True)
    else:
        st.subheader(f"📈 Current View: {st.session_state.current_chart.upper()} Chart")
        
        chart_fig = None
        
        if st.session_state.current_chart == 'bar':
            chart_fig = create_bar_chart(filtered_data)
        elif st.session_state.current_chart == 'line':
            chart_fig = create_line_chart(filtered_data)
        elif st.session_state.current_chart == 'pie':
            chart_fig = create_pie_chart(filtered_data)
        elif st.session_state.current_chart == 'scatter':
            chart_fig = create_scatter_plot(filtered_data)
        elif st.session_state.current_chart == 'heatmap':
            chart_fig = create_heatmap(filtered_data)
        
        if chart_fig:
            if isinstance(chart_fig, plt.Figure):
                st.pyplot(chart_fig)
            else:
                st.plotly_chart(chart_fig, use_container_width=True)
    
    # Data preview
    st.markdown("---")
    with st.expander("📋 View Data Table", expanded=False):
        st.dataframe(filtered_data, use_container_width=True)
        
        # Download button
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Filtered Data as CSV",
            csv,
            "filtered_data.csv",
            "text/csv",
            key='download-csv'
        )

if __name__ == "__main__":
    main()