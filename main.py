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
import pyaudio
from io import BytesIO
import time
from threading import Thread
import queue

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

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load sample data
@st.cache_data
def load_data():
    """Generate sample sales data"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=12, freq='ME')
    
    data = pd.DataFrame({
        'Month': dates.strftime('%B'),
        'Sales': np.random.randint(1000, 5000, 12),
        'Revenue': np.random.randint(2000, 8000, 12),
        'Profit': np.random.randint(500, 3000, 12),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 12),
        'Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 12)
    })
    return data

def detect_gesture(frame):
    """Detect hand gestures using MediaPipe"""
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
            
            # Thumb tip and index tip
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            
            # Count extended fingers
            fingers_up = 0
            if index_tip.y < landmarks[6].y: fingers_up += 1
            if middle_tip.y < landmarks[10].y: fingers_up += 1
            if ring_tip.y < landmarks[14].y: fingers_up += 1
            if pinky_tip.y < landmarks[18].y: fingers_up += 1
            
            # Detect gestures
            if fingers_up == 1:  # One finger - Next chart
                gesture = "next_chart"
            elif fingers_up == 2:  # Two fingers - Previous chart
                gesture = "previous_chart"
            elif fingers_up == 3:  # Three fingers - Apply filter
                gesture = "apply_filter"
            elif fingers_up == 4:  # Four fingers - Remove filter
                gesture = "remove_filter"
            elif fingers_up == 0:  # Fist - Comparison mode
                gesture = "toggle_comparison"
    
    return frame, gesture

def process_voice_command():
    """Process voice commands using SpeechRecognition"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
        command = recognizer.recognize_google(audio).lower()
        return command
    except sr.WaitTimeoutError:
        return "timeout"
    except sr.UnknownValueError:
        return "not_understood"
    except Exception as e:
        return f"error: {str(e)}"

def execute_voice_command(command):
    """Execute the recognized voice command"""
    if 'bar' in command or 'bar chart' in command:
        st.session_state.current_chart = 'bar'
        return "✅ Switched to Bar Chart"
    elif 'line' in command or 'line chart' in command:
        st.session_state.current_chart = 'line'
        return "✅ Switched to Line Chart"
    elif 'pie' in command or 'pie chart' in command:
        st.session_state.current_chart = 'pie'
        return "✅ Switched to Pie Chart"
    elif 'scatter' in command or 'scatter plot' in command:
        st.session_state.current_chart = 'scatter'
        return "✅ Switched to Scatter Plot"
    elif 'heatmap' in command or 'heat map' in command:
        st.session_state.current_chart = 'heatmap'
        return "✅ Switched to Heatmap"
    elif 'filter high' in command or 'high values' in command:
        st.session_state.filter_applied = 'high'
        return "✅ Filtered for high values"
    elif 'filter low' in command or 'low values' in command:
        st.session_state.filter_applied = 'low'
        return "✅ Filtered for low values"
    elif 'remove filter' in command or 'show all' in command or 'clear filter' in command:
        st.session_state.filter_applied = 'all'
        return "✅ Filter removed"
    elif 'compare' in command or 'comparison' in command:
        st.session_state.comparison_mode = not st.session_state.comparison_mode
        return f"✅ Comparison mode {'enabled' if st.session_state.comparison_mode else 'disabled'}"
    else:
        return "❌ Command not recognized. Try: 'show bar chart', 'filter high values', 'compare charts'"

def apply_filter(data, filter_type):
    """Apply filter to the dataset"""
    if filter_type == 'high':
        return data[data['Sales'] > data['Sales'].median()]
    elif filter_type == 'low':
        return data[data['Sales'] <= data['Sales'].median()]
    else:
        return data

def create_bar_chart(data):
    """Create bar chart using Plotly"""
    fig = px.bar(
        data, 
        x='Month', 
        y=['Sales', 'Revenue', 'Profit'],
        title='Monthly Sales, Revenue, and Profit',
        barmode='group',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
    )
    fig.update_layout(height=500, hovermode='x unified')
    return fig

def create_line_chart(data):
    """Create line chart using Plotly"""
    fig = px.line(
        data, 
        x='Month', 
        y=['Sales', 'Revenue', 'Profit'],
        title='Trend Analysis: Sales, Revenue, and Profit',
        markers=True
    )
    fig.update_layout(height=500, hovermode='x unified')
    return fig

def create_pie_chart(data):
    """Create pie chart using Plotly"""
    region_sales = data.groupby('Region')['Sales'].sum().reset_index()
    fig = px.pie(
        region_sales, 
        values='Sales', 
        names='Region',
        title='Sales Distribution by Region',
        hole=0.3
    )
    fig.update_layout(height=500)
    return fig

def create_scatter_plot(data):
    """Create scatter plot using Plotly"""
    fig = px.scatter(
        data, 
        x='Sales', 
        y='Profit',
        size='Revenue',
        color='Region',
        title='Sales vs Profit Analysis',
        hover_data=['Month', 'Category']
    )
    fig.update_layout(height=500)
    return fig

def create_heatmap(data):
    """Create heatmap using Seaborn"""
    fig, ax = plt.subplots(figsize=(10, 6))
    correlation_data = data[['Sales', 'Revenue', 'Profit']].corr()
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Heatmap')
    return fig

# Main App
def main():
    # Title and description
    st.title("📊 Multimodal Data Visualization System")
    st.markdown("**Control visualizations using gestures and voice commands**")
    st.markdown("---")
    
    # Load data
    data = load_data()
    filtered_data = apply_filter(data, st.session_state.filter_applied)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Manual chart selection
        st.subheader("Manual Selection")
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
        
        # Voice Control
        st.subheader("🎤 Voice Control")
        st.markdown("""
        **Available Commands:**
        - "Show bar chart"
        - "Show line chart"
        - "Show pie chart"
        - "Filter high values"
        - "Remove filter"
        - "Compare charts"
        """)
        
        if st.button("🎙️ Start Voice Command", use_container_width=True):
            command = process_voice_command()
            if command not in ['timeout', 'not_understood']:
                result = execute_voice_command(command)
                st.session_state.voice_command = f"Command: '{command}' - {result}"
            elif command == 'timeout':
                st.session_state.voice_command = "⏱️ Timeout - No speech detected"
            else:
                st.session_state.voice_command = "❌ Could not understand audio"
        
        if st.session_state.voice_command:
            st.info(st.session_state.voice_command)
        
        st.markdown("---")
        
        # Gesture Control
        st.subheader("✋ Gesture Control")
        st.markdown("""
        **Available Gestures:**
        - 1 finger: Next chart
        - 2 fingers: Previous chart
        - 3 fingers: Apply filter
        - 4 fingers: Remove filter
        - Fist: Toggle comparison
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
            st.multiselect(
                "Select charts to compare",
                ['bar', 'line', 'pie', 'scatter'],
                default=st.session_state.selected_charts,
                key='chart_comparison'
            )
            st.session_state.selected_charts = st.session_state.chart_comparison
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.metric("Total Records", len(data))
        st.metric("Filtered Records", len(filtered_data))
        st.metric("Filter Applied", st.session_state.filter_applied.upper())
    
    with col1:
        if st.session_state.filter_applied != 'all':
            st.info(f"🔍 Showing {st.session_state.filter_applied.upper()} values only")
    
    # Gesture recognition display
    if gesture_enabled:
        st.subheader("📹 Gesture Recognition")
        
        FRAME_WINDOW = st.empty()
        camera = cv2.VideoCapture(0)
        
        gesture_placeholder = st.empty()
        
        stop_button = st.button("Stop Gesture Recognition")
        
        while not stop_button:
            ret, frame = camera.read()
            if not ret:
                st.error("Cannot access camera")
                break
            
            frame, gesture = detect_gesture(frame)
            
            if gesture:
                st.session_state.gesture_detected = gesture
                
                # Execute gesture command
                if gesture == "next_chart":
                    charts = ['bar', 'line', 'pie', 'scatter', 'heatmap']
                    idx = charts.index(st.session_state.current_chart)
                    st.session_state.current_chart = charts[(idx + 1) % len(charts)]
                elif gesture == "previous_chart":
                    charts = ['bar', 'line', 'pie', 'scatter', 'heatmap']
                    idx = charts.index(st.session_state.current_chart)
                    st.session_state.current_chart = charts[(idx - 1) % len(charts)]
                elif gesture == "apply_filter":
                    st.session_state.filter_applied = 'high'
                elif gesture == "remove_filter":
                    st.session_state.filter_applied = 'all'
                elif gesture == "toggle_comparison":
                    st.session_state.comparison_mode = not st.session_state.comparison_mode
                
                gesture_placeholder.success(f"✋ Gesture Detected: {gesture.replace('_', ' ').title()}")
                time.sleep(1)
            
            FRAME_WINDOW.image(frame, channels="BGR")
        
        camera.release()
    
    st.markdown("---")
    
    # Display charts
    if st.session_state.comparison_mode and st.session_state.selected_charts:
        st.subheader("📊 Chart Comparison Mode")
        cols = st.columns(len(st.session_state.selected_charts))
        
        for idx, chart in enumerate(st.session_state.selected_charts):
            with cols[idx]:
                st.markdown(f"**{chart.upper()} Chart**")
                if chart == 'bar':
                    st.plotly_chart(create_bar_chart(filtered_data), use_container_width=True)
                elif chart == 'line':
                    st.plotly_chart(create_line_chart(filtered_data), use_container_width=True)
                elif chart == 'pie':
                    st.plotly_chart(create_pie_chart(filtered_data), use_container_width=True)
                elif chart == 'scatter':
                    st.plotly_chart(create_scatter_plot(filtered_data), use_container_width=True)
    else:
        st.subheader(f"📈 {st.session_state.current_chart.upper()} Chart")
        
        if st.session_state.current_chart == 'bar':
            st.plotly_chart(create_bar_chart(filtered_data), use_container_width=True)
        elif st.session_state.current_chart == 'line':
            st.plotly_chart(create_line_chart(filtered_data), use_container_width=True)
        elif st.session_state.current_chart == 'pie':
            st.plotly_chart(create_pie_chart(filtered_data), use_container_width=True)
        elif st.session_state.current_chart == 'scatter':
            st.plotly_chart(create_scatter_plot(filtered_data), use_container_width=True)
        elif st.session_state.current_chart == 'heatmap':
            st.pyplot(create_heatmap(filtered_data))
    
    # Data preview
    with st.expander("📋 View Data"):
        st.dataframe(filtered_data, use_container_width=True)
        
        # Download button
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Data as CSV",
            csv,
            "filtered_data.csv",
            "text/csv",
            key='download-csv'
        )

if __name__ == "__main__":
    main()