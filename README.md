**🩺 PCOS Detection Using Ultrasound Images <br><br>
📌 Overview** <br>
Polycystic Ovary Syndrome (PCOS) is a prevalent endocrine disorder among women of reproductive age. Accurate and timely diagnosis is crucial for effective management. This project leverages deep learning techniques to develop an automated system for detecting PCOS using ultrasound images, aiming to assist healthcare professionals in making informed decisions.

**🚀 Approach & Technologies Used <br><br>
🧠 Machine Learning & Deep Learning Models** <br>

**Transfer Learning:** Utilized pre-trained models such as VGG16 and ResNet to enhance feature extraction and improve model performance.

**XGBoost:** Implemented as a classifier to boost the accuracy of PCOS detection.

**🗂 Dataset** <br>
**Data Collection**: The dataset comprises ultrasound images labeled as 'PCOS' and 'Non-PCOS.' Due to the scarcity of publicly available data, image augmentation techniques were applied to expand the dataset and improve model generalization.

**Preprocessing Steps:** <br>

**Salt Segmentation:** Applied to enhance the contrast of ultrasound images, facilitating better differentiation of ovarian structures.

**Otsu Thresholding:** Employed to automatically determine the optimal threshold value for segmenting the image into foreground (potential cysts) and background, aiding in accurate cyst detection. 

Resizing images to a uniform dimension suitable for model input.

Normalization to standardize pixel values.

Segmentation techniques to highlight relevant ovarian features.

**📊 Model Training & Evaluation**
Training Process: Models were trained on the preprocessed dataset, with hyperparameters tuned for optimal performance.

**Evaluation Metrics:**

✅ Accuracy

✅ Precision

✅ Recall

✅ F1-score

These metrics provide a comprehensive assessment of the model's performance in detecting PCOS.

**🔍 Key Findings & Challenges** <br>
**Performance:** The combination of CNNs with transfer learning and XGBoost demonstrated high accuracy in classifying PCOS and Non-PCOS images.

**Challenges:** Limited availability of labeled ultrasound images necessitated the use of data augmentation. Variations in image quality and differences in ultrasound machines introduced additional complexity.

**🔧 Installation**
To set up the PCOS Detection project locally:

# Clone the repository
git clone https://github.com/SSHarshitha/PCOS_Detection-using-ultrasound-images.git <br>
cd PCOS_Detection-using-ultrasound-images

# Install dependencies
pip install -r requirements.txt <br><br>
**🚀 Usage <br>**<br>
After installation, you can utilize the trained model to detect PCOS from ultrasound images. <br>
Here's a basic example:

python
Copy
Edit
from model import predict_pcos
from PIL import Image

# Load an ultrasound image
image = Image.open('path_to_image.jpg')

# Prediction
result = predict_pcos(image)
print(f"Prediction: {'PCOS Detected' if result else 'No PCOS Detected'}")
