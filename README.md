# 😷 Face Mask Detection System

A deep learning-based **Face Mask Detection System** that detects human faces from images or camera input and classifies whether each person is **wearing a mask or not wearing a mask**.

The system uses **MobileNetV2** for image classification and **OpenCV Haar Cascade** for face detection. A Flask backend handles the machine learning inference, while the React frontend provides an interactive user interface.

---

## 🚀 Features

* 😷 Detects **Mask / No Mask**
* 👤 Detects multiple faces in an image
* 📊 Displays prediction confidence
* 🟩 Draws bounding boxes around detected faces
* ⚡ Fast prediction using MobileNetV2
* 🖼️ Image upload functionality
* 📷 Supports camera-based detection
* 🎨 Modern React + Tailwind CSS interface
* 🔌 Flask REST API backend
* 🐳 Docker support
* 📱 Responsive user interface

---

## 🛠️ Technologies Used

### Machine Learning

* Python
* TensorFlow
* Keras
* MobileNetV2
* OpenCV
* NumPy
* Pandas
* Matplotlib

### Backend

* Flask
* Flask-CORS
* Python
* REST API

### Frontend

* React.js
* Tailwind CSS
* JavaScript
* HTML5
* CSS3

### Deployment

* Docker
* Docker Compose
* Git & GitHub

---

## 🧠 System Architecture

```text
                    ┌─────────────────────┐
                    │      User           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   React Frontend    │
                    │   + Tailwind CSS    │
                    └──────────┬──────────┘
                               │
                         HTTP Request
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Flask Backend    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   OpenCV Haar       │
                    │   Face Detection    │
                    └──────────┬──────────┘
                               │
                         Face Cropping
                               │
                               ▼
                    ┌─────────────────────┐
                    │    MobileNetV2      │
                    │   Classification    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Mask / No Mask +    │
                    │ Confidence Score    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   React Frontend    │
                    │ Prediction Display  │
                    └─────────────────────┘
```

---

## 📂 Project Structure

```text
Face-Mask-Detection-System/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── App.jsx
│   │   └── main.jsx
│   │
│   ├── public/
│   ├── package.json
│   └── tailwind.config.js
│
├── backend/
│   ├── app.py
│   ├── best_model.h5
│   ├── haarcascade_frontalface_default.xml
│   ├── requirements.txt
│   └── uploads/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

> Adjust the folder names above if your actual GitHub repository uses a different structure.

---

# 📊 Dataset

The model is trained using a face mask image dataset containing two classes:

```text
Dataset
│
├── With Mask
│
└── Without Mask
```

### Classes

| Class     | Description                       |
| --------- | --------------------------------- |
| `Mask`    | Person is wearing a face mask     |
| `No Mask` | Person is not wearing a face mask |

Before training, images are resized to **224 × 224 pixels** to match the MobileNetV2 input requirements.

---

# 🧠 Model

## MobileNetV2

The project uses **MobileNetV2**, a lightweight convolutional neural network designed for efficient image classification.

MobileNetV2 is suitable for this application because it provides a good balance between:

* Accuracy
* Speed
* Model size
* Computational efficiency

### Model Pipeline

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
Face Detection
     │
     ▼
Face Cropping
     │
     ▼
Resize to 224 × 224
     │
     ▼
MobileNetV2
     │
     ▼
Classification
     │
     ├── Mask
     │
     └── No Mask
     │
     ▼
Confidence Score
```

---

# 🔍 Face Detection

The system uses the **Haar Cascade Classifier** from OpenCV to detect faces.

```python
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)
```

Once faces are detected:

1. Bounding boxes are created.
2. Each face is cropped.
3. The cropped face is resized.
4. The MobileNetV2 model performs classification.
5. The prediction and confidence are displayed.

---

# 📈 Prediction Output

For each detected face, the system provides:

```text
Face 1
Prediction: Mask
Confidence: 96.42%
```

or

```text
Face 1
Prediction: No Mask
Confidence: 98.15%
```

For multiple people:

```text
Face 1 → Mask     → 96.42%
Face 2 → No Mask  → 98.15%
Face 3 → Mask     → 91.73%
```

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Face-Mask-Detection-System.git
```

Navigate into the project:

```bash
cd Face-Mask-Detection-System
```

---

# 🐍 Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

The backend will run on:

```text
http://127.0.0.1:5000
```

---

# ⚛️ Frontend Setup

Open another terminal and navigate to the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the React development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173
```

---

# 🐳 Run Using Docker

Build and start the containers:

```bash
docker-compose up --build
```

To stop the containers:

```bash
docker-compose down
```

---

# 📷 How to Use

### Step 1

Open the web application.

### Step 2

Upload an image containing one or more faces.

### Step 3

The backend receives the image.

### Step 4

OpenCV detects the faces.

### Step 5

Each detected face is passed to the trained MobileNetV2 model.

### Step 6

The application displays:

* Face bounding box
* Mask / No Mask prediction
* Confidence score

---

# 🔌 API

The Flask backend provides an API endpoint for prediction.

### Prediction Endpoint

```text
POST /predict
```

The frontend sends an image to the backend.

Example request:

```text
POST /predict
Content-Type: multipart/form-data
```

Example response:

```json
{
    "prediction": "Mask",
    "confidence": 96.42
}
```

> Modify the endpoint and response format in this README if your actual Flask API uses different names.

---

# 📸 Application Workflow

```text
Upload Image
     ↓
Flask API
     ↓
OpenCV Face Detection
     ↓
Crop Detected Faces
     ↓
Image Preprocessing
     ↓
MobileNetV2 Prediction
     ↓
Mask / No Mask
     ↓
Confidence Calculation
     ↓
Display Result
```

---

# 🎯 Use Cases

This project can be used as a foundation for:

* 🏥 Healthcare environments
* 🏢 Office buildings
* 🏫 Educational institutions
* 🏭 Industrial workplaces
* 🚉 Public transportation
* 🛍️ Shopping malls
* 🏪 Public facilities
* 🔐 Access-control systems

---

# 🔮 Future Improvements

The project can be further improved by adding:

* Real-time webcam detection
* YOLO-based face detection
* Improved model accuracy
* Face tracking
* Real-time monitoring dashboard
* Database storage for predictions
* User authentication
* Prediction history
* Cloud deployment
* Mobile application
* Email/SMS alerts
* Improved performance for low-light images

---

# ⚠️ Limitations

* Detection performance can decrease with poor lighting.
* Extremely small faces may not be detected correctly.
* Face occlusion can affect detection.
* Prediction accuracy depends on the training dataset.
* Haar Cascade may be less robust than modern object-detection models.

---

# 📌 Project Highlights

### Machine Learning

* Transfer Learning
* MobileNetV2
* Image Classification
* Data Augmentation
* Computer Vision

### Backend

* Python
* Flask
* REST API
* OpenCV

### Frontend

* React.js
* Tailwind CSS
* Responsive UI

### DevOps

* Docker
* Docker Compose
* Git
* GitHub

---

# 👩‍💻 Author

**Ruthu Madhavi**

### GitHub

https://github.com/Ruthu543

---

# ⭐ Support

If you found this project useful, please consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is intended for educational and demonstration purposes.
