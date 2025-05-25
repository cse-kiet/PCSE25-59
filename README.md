# PCSE25-59

*Team Members:*

* *Vikas Kumar* (21002901000188 – CSE – 8C)
* *Shreyansh Tiwari* (21002901000159 – CSE – 8C)


*Project Guide:*
*Mr. Rahul Kumar Sharma* (Associate Professor, CSE)

---

## Deep TrafficFlow: Deep Learning-Based Real-Time Traffic Congestion Management

### Overview

*Deep TrafficFlow* is an intelligent traffic monitoring and congestion prediction system built using deep learning and computer vision. The system leverages video footage to detect, classify, and track vehicles in real-time. Using a hybrid of Fast R-CNN and motion tracking, it provides accurate traffic flow analysis and congestion prediction with high precision.

This project was developed as part of the Bachelor of Technology curriculum and has been presented at academic conferences.

---

### Core Technologies Used

* *Fast R-CNN*: For vehicle detection and classification.
* *OpenCV*: For video processing and motion-based vehicle tracking.
* *Python (TensorFlow, Keras, NumPy, Pandas)*: For model development and data handling.
* *Scikit-learn*: For congestion prediction using ML classifiers like Decision Trees, Random Forest, and SVM.
* *Matplotlib/Seaborn*: For visualization of results and predictions.
* *Flask*: For optional web deployment and API integration.

---

### Features

* *Real-Time Vehicle Detection*: Detects vehicles using a trained Fast R-CNN model.
* *Vehicle Classification*: Categorizes detected vehicles (e.g., car, bus, bike).
* *Motion Path Tracking*: Tracks vehicle movements frame-by-frame for trajectory analysis.
* *Congestion Prediction*: Utilizes traffic patterns and ML classifiers to predict congestion likelihood.
* *Analytics Dashboard*: Optional visualization of traffic metrics and predictions.

---

### Project Structure
```
PCSE25-59/
├── congestion_model/
│   ├── fast_rcnn_model.h5         # Pretrained Fast R-CNN model
│   ├── motion_tracking.py         # Path tracking using OpenCV
│   ├── congestion_predictor.py   # ML model for congestion prediction
│   ├── utils.py                  # Helper functions
│   ├── requirements.txt          # Python dependencies
│   └── traffic_video.mp4         # Sample video input
├── documents/
│   ├── project_report.pdf        # Complete project documentation
│   ├── synopsis.pdf              # Project synopsis
│   ├── certificate_project_report.pdf  # Certificate signed by project guide
│   ├── plag_report.pdf           # Plagiarism report
│   └── Journal_documents/
│       ├── ResearchPaper-2025.pdf      # Research paper for ICETISD-2025
│       ├── Presentation_certificate_Vikas_Kumar.pdf # Presentation certificate
│       └── Deep_TrafficFlow_Presentation.pptx  # Conference presentation
└── .gitignore                     # Git ignore file to exclude cache and secrets
```

### How It Works

1. *Input*: Real-time or pre-recorded traffic footage.
2. *Detection*: Fast R-CNN identifies vehicles frame-by-frame.
3. *Tracking*: A motion tracker (based on background subtraction and contour mapping) maps vehicle paths.
4. *Classification & Counting*: Each vehicle is counted and classified using CNN layers.
5. *Congestion Prediction*: Statistical and ML-based analysis of vehicle count, speed, and density is used to predict traffic congestion probability.
6. *Output*: A congestion level is displayed or logged, optionally visualized via Flask UI.



### Prerequisites

* Python 3.10+
* pip (Python package manager)
* NVIDIA GPU (GTX 1650 or higher recommended for real-time inference)
* Required Python libraries listed in requirements.txt



### Setup and Installation

*1. Clone the Repository*
```
git clone https://github.com/cse-kiet/PCSE25-59.git
cd PCSE25-59/congestion_model
```


*2. Create a Virtual Environment (Recommended)*
```
python -m venv venv
```
# Activate environment
# Windows
```
venv\Scripts\activate

```
# macOS/Linux
```
source venv/bin/activate
```

*3. Install Dependencies*

```
pip install -r requirements.txt
```

*4. Run the System*

To test the model on sample video:

```
python car-object-detection.ipynb

```
To predict congestion based on extracted features:

```
python car-speed-detection-using-fastcnn.ipynb
```
```
python car-video-path-tracking.ipynb
```
To run the main app
```
python app.py
```


### Included Documents

Located in the documents/ folder:

* project_report.pdf: Complete documentation of the project.
* synopsis.pdf: Project synopsis.
* certificate_project_report.pdf: Certificate signed by project guide.
* plag_report.pdf: Plagiarism report.
* conference_documents/: Contains the research paper, presentation, and certificates related to ICETISD-2025.

```

### .gitignore
```
Common Python cache files and secret environment configurations are excluded from version control.

```
__pycache__/
*.pyc
venv/
.env

```
