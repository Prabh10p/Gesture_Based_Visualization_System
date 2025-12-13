# 📊 Multimodal Data Visualization System

An interactive **Streamlit-based multimodal data visualization application** that allows users to explore data using **manual controls, voice commands, and hand gestures**. The system supports multiple chart types, filtering, comparison mode, and CSV uploads, combining data analytics with computer vision and speech recognition.

# 🚀 Features

### 🔹 Data Sources

* Built-in **sample dataset** (sales, revenue, profit by month, region, and category)
* Upload your own **CSV file** for custom analysis

### 🔹 Visualization Types

* 📊 Bar Chart
* 📈 Line Chart
* 🥧 Pie Chart
* 🔵 Scatter Plot
* 🔥 Correlation Heatmap

### 🔹 Interaction Modes

#### 🎛️ Manual Controls

* Select chart type
* Apply data filters (All / High / Low)
* Enable comparison mode

#### 🎤 Voice Control (Speech Recognition)

Control the app using natural language commands:

* "Show bar chart"
* "Show line chart"
* "Show pie chart"
* "Show scatter plot"
* "Show heatmap"
* "Filter high values"
* "Filter low values"
* "Remove filter"
* "Compare charts"

#### ✋ Gesture Control (Computer Vision)

Uses **MediaPipe + OpenCV** for real-time hand gesture recognition:

| Gesture        | Action                  |
|----------------|-------------------------|
| 👆 1 finger    | Next chart              |
| ✌️ 2 fingers   | Previous chart          |
| 🤟 3 fingers   | Apply high-value filter |
| 🖖 4 fingers   | Remove filter           |
| ✊ Fist (0)     | Toggle comparison mode  |
| 🖐️ 5 fingers  | Show all data           |


### 🔹 Comparison Mode

* View multiple chart types **side-by-side**
* Useful for pattern comparison and exploratory analysis

### 🔹 Data Utilities

* Filtered dataset preview
* Download filtered data as CSV

---

## 🧠 Technologies Used

| Category           | Tools / Libraries              |
| ------------------ | ------------------------------ |
| Web App            | Streamlit                      |
| Data Processing    | Pandas, NumPy                  |
| Visualization      | Plotly, Matplotlib, Seaborn    |
| Computer Vision    | OpenCV, MediaPipe              |
| Speech Recognition | SpeechRecognition (Google API) |

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/multimodal-data-visualization.git
cd multimodal-data-visualization
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Packages:**

* streamlit
* pandas
* numpy
* plotly
* matplotlib
* seaborn
* opencv-python
* mediapipe
* SpeechRecognition
* pyaudio (or alternative mic backend)

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

> ⚠️ Make sure camera and microphone permissions are enabled for gesture and voice control.

---

## 📁 Project Structure

```
├── app.py                  # Main Streamlit application
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
```

---

## 📊 Sample Dataset

The built-in dataset simulates monthly sales data:

* Month
* Sales
* Revenue
* Profit
* Region
* Category

This allows immediate exploration without uploading data.

---

## ⚠️ Known Limitations

* Gesture recognition accuracy may vary based on lighting and camera quality
* Voice recognition requires an active internet connection
* Heatmap requires at least two numeric columns

---

## 🌱 Future Enhancements

* Multi-hand gesture support
* Custom gesture mapping
* NLP-based voice commands
* Dashboard export (PDF / PNG)
* User authentication & session saving

---

## 👨‍💻 Author

**Prabhjot Singh**
Data Analytics & Visualization Enthusiast

---

## 📜 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute.

---

⭐ *If you find this project useful, consider giving it a star!*
