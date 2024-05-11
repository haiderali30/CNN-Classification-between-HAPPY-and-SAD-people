# CNN-Classification-between-HAPPY-and-SAD-people
This project utilizes Convolutional Neural Networks (CNN) to classify facial expressions between happiness and sadness. The CNN model is trained on a dataset containing images of individuals exhibiting these emotions. The project aims to accurately identify the emotional states of individuals based on their facial expressions.

# Installation
Ensure you have the necessary dependencies installed:

pip install tensorflow tensorflow-gpu opencv-python matplotlib

# Dataset
The dataset consists of images categorized into "happy" and "sad" folders within the images directory. Images not conforming to standard image extensions are filtered out. Preprocessing steps include resizing images to a uniform size and normalizing pixel values.

# Model Architecture
The CNN model comprises several convolutional and pooling layers followed by fully connected layers. The architecture is designed to extract features from input images and learn patterns associated with happy and sad expressions.

# Training
The model is trained using the fit method on a split dataset containing training and validation sets. Training progress and performance metrics such as loss and accuracy are monitored using TensorBoard.

# Evaluation
Performance evaluation includes metrics such as precision, recall, and accuracy computed on a separate test set. Additionally, individual images can be fed into the trained model for real-time prediction of emotional states.

# Saving the Model
Once trained, the model is saved in the HDF5 format for future use. The saved model can be loaded and used for inference on new images to predict emotional states.


