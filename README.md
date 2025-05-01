# 🩺 Skin Disease Classification (10 Classes)

This project aims to classify skin diseases into 10 categories using deep learning and transfer learning with a pre-trained EfficientNet-B0 model. It includes data preprocessing, model training, evaluation, and a FastAPI deployment for inference.

## 📊 Dataset

We used the publicly available [Skin Diseases Image Dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset), which contains labeled images of 10 skin disease categories.

## 🔄 Data Preprocessing

Effective preprocessing is crucial for enhancing model performance. The following steps were undertaken:

- **Dataset Statistics Calculation**: Computed the actual mean and standard deviation of the dataset to normalize the images accurately.

- **Data Augmentation for Training**:

  - _Resizing_: Adjusted all images to a uniform size of 224x224 pixels to match the input dimensions expected by _EfficientNet-B0_.
  - _Random Affine Transformations_: Applied random scaling (between 85% to 110%), translation (up to 10% shift), and rotation (±10 degrees) to simulate various image orientations and scales.
  - _Color Jittering_: Introduced slight variations in brightness, contrast, saturation, and hue to account for different lighting conditions.
  - _Random Flipping_: Performed horizontal and vertical flips with a 50% probability to augment the dataset and prevent overfitting.
  - _Normalization_: Standardized the pixel values using the previously computed mean and standard deviation.

- **Preprocessing for Testing and Validation**:

  - _Resizing_: Adjusted images to 224x224 pixels.
  - _Normalization_: Applied the same normalization as used in training to maintain consistency.

- Created separate **train**, **test**, and **validation** dataloaders.
- **Class Label Cleaning**: Simplified class names by removing extraneous information. For instance, "6. Benign Keratosis-like Lesions (BKL) 2624" was renamed to "Benign Keratosis". The cleaned class mappings were saved in an `idx_to_class.json` file for reference.

## 🧠 Model Architecture

- **Base Model**: Leveraged the pre-trained EfficientNet-B0 model from `torchvision.models` for its balance between performance and computational efficiency.

- **Modifications**:

  - Replaced the final fully connected layer to output predictions for 10 classes, aligning with our dataset.

- **Loss Function**: Employed CrossEntropyLoss, suitable for multi-class classification tasks.

- **Optimizer**: Utilized the Adam optimizer for its adaptive learning rate capabilities.

- **Learning Rate Scheduler**: Implemented an ExponentialLR scheduler to progressively decrease the learning rate, aiding in model convergence.

## 🚀 Training

- **Epochs**: Trained the model for 10 epochs.

- **Performance**: Achieved approximately 80% accuracy on both validation and test datasets.

- **Model Saving**: The trained model was saved as `model.pth` in the project directory for future inference.

## 🖥️ Inference API

A FastAPI application was developed to facilitate interaction with the trained model. Users can upload an image and receive the predicted disease class in response.

## 🧪 Setup Instructions

### 1. Create a new environment

```bash
conda create -n skin-disease python=3.10.12
conda activate skin-disease
```

### 2. Clone the project and install dependencies

```bash
cd Skin-disease/src/
pip install -r requirements.txt
```

### 3. Train the model (optional)

- Run the script:

```bash
python training.py
```

- Or use the Jupyter notebook:
  Download and run `training_notebook.ipynb` on **Colab** or **Kaggle Notebooks**.

### 4. Run the FastAPI server

```bash
uvicorn main:app --host '0.0.0.0' --port 5000 --reload
```

- You can use [Postman](https://www.postman.com/) to test the API using the provided `skin-disease.postman_collection.json` file.

## 📬 Feedback

If you have any feedback, suggestions, or improvements, feel free to open an issue or submit a pull request.
