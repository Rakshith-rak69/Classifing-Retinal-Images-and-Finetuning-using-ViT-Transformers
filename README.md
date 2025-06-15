# **Classifying Retinal Images for Diabetic Retinopathy using Vision Transformers**

This project focuses on developing and evaluating deep learning models for the automated classification of retinal fundus images to detect Diabetic Retinopathy (DR). It explores the effectiveness of both Convolutional Neural Networks (CNNs) and state-of-the-art [Vision Transformers (ViT)](https://huggingface.co/google/vit-base-patch16-224) in this medical imaging task, with a particular emphasis on improving generalization and performance on minority classes.

## **✨ Key Features**

* **Diabetic Retinopathy Classification:** Automated detection of DR from retinal fundus images.  
* **Comparative Model Analysis:** Comprehensive comparison between CNN (EfficientNetB0) and [Vision Transformer (ViT)](https://huggingface.co/google/vit-base-patch16-224) architectures.  
* **Advanced Fine-tuning:** Implementation of fine-tuning techniques for pre-trained ViT models to optimize performance on medical images.  
* **Robust Data Preprocessing & Augmentation:** Utilizes TensorFlow for efficient image handling and augmentation to enhance model generalization.  
* **Focus on Minority Class Performance:** Specific attention to improving detection rates for non-DR cases, which can often be a challenge in imbalanced datasets.  
* **Detailed Performance Evaluation:** Thorough assessment using key metrics including Accuracy, Precision, Recall, and F1-score.

## **📊 Performance Highlights**

The project demonstrated a significant leap in classification performance with Vision Transformers:

* **CNN Baseline:** Achieved **58% accuracy**.  
* **Fine-tuned Vision Transformer (ViT):** Achieved **81% accuracy**, representing a **23% performance improvement** over the CNN baseline.  
* **Superior Generalization for Minority Class (non-DR)::**  
  * Precision increased from **0.24 to 0.69**.  
  * Recall increased from **0.12 to 0.74**.  
* **Enhanced Clinical Reliability:** Significant reduction in false negatives and false positives, crucial for real-world medical applications.

## **🚀 Technologies Used**

* **Deep Learning Frameworks:** PyTorch  
* **Model Architectures:** Convolutional Neural Network (CNN), [Vision Transformer (ViT)](https://huggingface.co/google/vit-base-patch16-224)  
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Hugging Face  
* **Dataset:** [IDRiD (Indian Diabetic Retinopathy Image Dataset)](https://idrid.grand-challenge.org/Data/)

## **💻 How to Run (General Steps)**

1. Set Up Environment:

```bash
It's recommended to use a virtual environment.  
conda create \-n dr\_env python=3.9  
conda activate dr\_env  
\# or  
python \-m venv dr\_env  
source dr\_env/bin/activate  
```

2. **Install Dependencies:**

```bash
pip install \-r requirements.txt
```

*(Note: A requirements.txt file is typical for listing all Python dependencies.)*

3. Download Dataset:  
   Obtain the IDRiD dataset from its official source (often requires registration or agreement to terms of use). Place the images and labels in a designated data/ directory.  
4. Run Jupyter Notebook/Scripts:  
   Execute the provided Jupyter notebooks or Python scripts to preprocess data, train models, and evaluate performance.

## **📁 Dataset Information**

This project primarily utilizes the [**IDRiD (Indian Diabetic Retinopathy Image Dataset)**](https://idrid.grand-challenge.org/Data/), a publicly available dataset specifically curated for diabetic retinopathy detection and grading. It provides retinal fundus images along with corresponding ground truth labels for DR severity.

### **License Details**

The IDRiD dataset is licensed under a Creative Commons Attribution 4.0 International License.

(c) by Prasanna Porwal, Samiksha Pachade and Manesh Kokare.

You should have received a copy of the license along with this work. If not, see [http://creativecommons.org/licenses/by/4.0/](http://creativecommons.org/licenses/by/4.0/).

## **🚧 Future Work & Disclaimer**

Future work could explore:

* Utilizing different model architectures or more specialized pre-trained vision transformers.  
* Implementing advanced dataset balancing techniques.  
* Collecting and incorporating more diverse data.  
* Applying more rigorous testing and validation strategies.

**Disclaimer:** These models are developed for research and educational purposes and are not intended for real-world medical imaging applications. They should not be relied upon for diagnosing, predicting, or managing diabetic retinopathy or any other health outcomes. Any health-related decisions should always be made with the guidance of a qualified healthcare professional.

Made with ❤️ by Rakshith
