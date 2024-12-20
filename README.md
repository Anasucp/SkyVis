
## 📄 **README.md**

```markdown
# 🌤️ SkyVis: Weather Classification Web App

SkyVis is a web-based application that uses a deep learning model to classify weather conditions from images. The model can predict four weather categories: **Cloudy, Rain, Sunrise, and Shine**. This project uses a Convolutional Neural Network (CNN) with the **MobileNetV2** architecture for image classification, and the web interface is built using **Flask** with **Bootstrap** for styling.

---

## 🚀 **Features**

- **Weather Classification**: Predicts weather categories based on uploaded images.
- **Web Interface**: User-friendly web app for uploading images and viewing predictions.
- **Responsive Design**: Built with Bootstrap for mobile and desktop compatibility.
- **Data Augmentation**: Enhances the dataset to improve model performance.
- **Visualization**: Displays training accuracy, loss plots, and confusion matrices for evaluation.

---

## 📂 **Project Structure**

```bash
SkyVis/
├── data/                          # Dataset folder with weather categories
│   ├── cloudy/                    # Images of cloudy weather
│   ├── rain/                      # Images of rainy weather
│   ├── sunrise/                   # Images of sunrise weather
│   └── shine/                     # Images of shine (sunny) weather
├── scripts/                       # Python scripts for various tasks
│   ├── preprocess.py              # Script for preprocessing and augmenting the dataset
│   ├── train.py                   # Script for training the CNN model
│   └── predict.py                 # Script for making predictions with the trained model
├── models/                        # Folder to store trained models
│   └── weather_model.h5           # Trained model file
├── uploads/                       # Folder to store uploaded images via the web app
├── results/                       # Folder to store evaluation results
│   ├── accuracy_plot.png          # Training and validation accuracy plot
│   ├── loss_plot.png              # Training and validation loss plot
│   └── confusion_matrix.png       # Confusion matrix plot
├── templates/                     # HTML templates for Flask web app
│   ├── index.html                 # Home page template (file upload form)
│   └── result.html                # Result page template (shows prediction and image)
├── static/                        # Static files (CSS, images, etc.)
│   └── styles.css                 # Custom CSS for styling the web interface
├── app.py                         # Flask web application script
├── README.md                      # Detailed project documentation
└── requirements.txt               # List of project dependencies
```

---

## 🛠️ **Installation**

### **1. Clone the Repository**

```bash
git clone https://github.com/Anasucp/SkyVis.git
cd SkyVis
```

### **2. Set Up a Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**

Create a **`requirements.txt`** file with the following content:

```
Flask==2.0.1
tensorflow==2.6.0
numpy==1.19.5
pandas==1.3.3
matplotlib==3.4.3
seaborn==0.11.2
opencv-python==4.5.3.56
```

Then, install the dependencies:

```bash
pip install -r requirements.txt
```

### **4. Prepare the Dataset**

Ensure the dataset is organized in the `data/` folder as follows:

```bash
data/
├── cloudy/
├── rain/
├── sunrise/
└── shine/
```

Each folder should contain images relevant to that weather condition.

### **5. Preprocess the Data**

Run the preprocessing script to load and augment the dataset:

```bash
python3 scripts/preprocess.py
```

### **6. Train the Model**

Train the CNN model using the pre-trained MobileNetV2 architecture:

```bash
python3 scripts/train.py
```

The trained model will be saved in the `models/` folder.

### **7. Run the Web App**

Start the Flask web application:

```bash
python3 app.py
```

Visit the app in your browser at:  
`http://127.0.0.1:5000`

---

## 🌐 **Usage**

1. **Upload an Image**: Go to the home page and upload an image of the weather condition.
2. **Get Prediction**: The app will display the predicted weather category (`Cloudy`, `Rain`, `Sunrise`, or `Shine`).
3. **Try Again**: Use the button to upload another image and get a new prediction.

---

## 📊 **Model Evaluation**

### **Accuracy and Loss Plots**

During training, accuracy and loss plots are saved in the `results/` folder:

- **`accuracy_plot.png`**: Shows training and validation accuracy.
- **`loss_plot.png`**: Shows training and validation loss.

### **Confusion Matrix**

The confusion matrix is saved as **`confusion_matrix.png`** in the `results/` folder. This helps visualize the model's performance across different categories.

---

## 🎨 **Web Interface**

### **Home Page**

- **File Upload**: Upload an image for prediction.
- **Styled with Bootstrap**: Responsive design for mobile and desktop.

### **Result Page**

- **Prediction Display**: Shows the predicted category.
- **Image Preview**: Displays the uploaded image.
- **Retry Button**: Easily upload another image.

---

## 🐞 **Troubleshooting**

### **Common Issues**

1. **Issue**: `ModuleNotFoundError: No module named 'seaborn'`  
   **Solution**: Install `seaborn` using:  
   ```bash
   pip install seaborn
   ```

2. **Issue**: Image not displaying on the result page.  
   **Solution**: Ensure `app.py` correctly saves images in the `uploads/` folder and serves them via the `/uploads/<filename>` route.

3. **Issue**: CUDA-related warnings when running TensorFlow.  
   **Solution**: Force TensorFlow to use CPU by adding this at the top of your script:  
   ```python
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
   ```

---

## 📜 **License**

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute it as you wish.

---

## 🙌 **Contributions**

Contributions are welcome! If you'd like to improve this project, please submit a pull request.

### **Steps to Contribute**

1. **Fork the repository**.
2. **Create a new branch**:  
   ```bash
   git checkout -b feature-branch
   ```
3. **Make your changes** and commit them:  
   ```bash
   git commit -m 'Add new feature'
   ```
4. **Push to the branch**:  
   ```bash
   git push origin feature-branch
   ```
5. **Open a pull request**.

---

## 📞 **Contact**

- **Author**: Muhammad Ans  
- **Email**: [anas.ccj420@gmail.com]  
- **GitHub**: [https://github.com/Anasucp](https://github.com/Anasucp)

---

