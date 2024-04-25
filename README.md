# Breast Cancer Classification with Neural Network

# Overview
This project aims to develop a neural network-based classifier to predict whether a breast cancer cell is Malignant or Benign. The model utilizes data preprocessing techniques and neural network architecture implemented using TensorFlow and Keras libraries.

# Dataset
The dataset used for training and testing the classifier is the Breast Cancer Wisconsin (Diagnostic) dataset, which is publicly available and contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The dataset includes various attributes such as mean radius, mean texture, mean perimeter, and more.

# Approach
Data Preprocessing: The dataset underwent preprocessing steps, including handling missing values, encoding categorical variables, and scaling numerical features using the StandardScaler function.
Neural Network Architecture: A neural network model with three layers was constructed using TensorFlow and Keras. The architecture consists of input, hidden, and output layers, with appropriate activation functions and neuron units.
Model Training: The preprocessed data was split into training and validation sets. The model was trained using the training data, and its performance was evaluated using the validation set.
Evaluation: The model's performance metrics such as loss and accuracy were monitored during training.

# Challenges
Overfitting: Initially, the model exhibited signs of overfitting, as indicated by increasing loss on the training set while validation loss remained constant. This issue was addressed by reducing the number of training epochs.

# Solution
Reduced Epochs: By reducing the number of training epochs from the initial value, the overfitting problem was mitigated. This led to improved model generalization performance, as evidenced by decreasing loss and increasing accuracy on both training and validation sets.


<img src="breast cancer/Screenshot 2024-04-25 at 10.27.32.png" alt="Model Accuracy Graph">



# Results
Model Accuracy: The trained model achieved a high accuracy score of 99.5% on the training set and 97.8% on the validation set, demonstrating its effectiveness in classifying breast cancer cells.

# Recommendation
Hyperparameter Tuning: Further optimization of hyperparameters such as learning rate, batch size, and network architecture could potentially enhance the model's performance.
Data Augmentation: Augmenting the dataset with additional samples or applying techniques like data augmentation may help improve model robustness and generalization.


