# Classification-with-Logistic-Regression-on-Iris-Dataset

## Overview
This repository focuses on solving a classification problem using logistic regression. The challenge is to correctly identify flower species from the well-known iris dataset, employing logistic regression techniques.

## Getting Started

### Dependencies
To install necessary libraries, run the following command:

```bash
pip install torch
pip install numpy
pip install sklearn
pip install pandas
pip install seaborn
pip install scikit-learn
pip install matplotlib
```
### Dataset
The project uses the Iris dataset, a classic in classification problems, available through scikit-learn.

### Concept Overview
The project tackles a classification problem utilizing logistic regression, a fundamental machine learning technique. Focused on the Iris dataset, the task is to classify iris flowers into one of three species based on four features: sepal length, sepal width, petal length, and petal width. This dataset is a classic in the field of machine learning and statistics, often used for testing classification algorithms.

Logistic regression, despite its name, is a linear model for classification rather than regression. It is used when the dependent variable is categorical. The algorithm predicts the probability of the target labels by applying a logistic function to a linear combination of the input features. This project demonstrates both binary classification (distinguishing one class from all others) and multiclass classification (distinguishing among three or more classes) using logistic regression.


### Features
- **Binary and Multiclass Classification**: Implements logistic regression to solve both binary (one-vs-all) and multiclass classification problems on the Iris dataset.
- **Custom Logistic Regression Implementation**: Features a logistic regression model built with PyTorch, showcasing the flexibility and power of this deep learning framework even for relatively simple tasks.
- **Data Preprocessing**: Includes normalization of features to improve model performance, along with detailed explanations of why and how preprocessing is conducted.
- **Visualization**: Utilizes matplotlib and seaborn for data visualization, providing insights into the dataset and the model's performance. Visualizations include scatter plots of the features, decision boundary plots, and loss curves.
- **Experimentation with Hyperparameters**: Explores the impact of different learning rates and activation functions on model accuracy and convergence, with the results discussed in the context of logistic regression's sensitivity to these parameters.
- **Model Evaluation**: Evaluates the model's performance on a test set, presenting accuracy as the primary metric. Additionally, explores the XOR problem to demonstrate logistic regression's limitations and potential need for non-linear models.


### Experiments and Results
The project involved several key experiments to evaluate the effectiveness of logistic regression in classifying iris species and to understand the impact of various hyperparameters and model configurations. Below is a summary of the experiments conducted and their results:

**1. Binary Classification Accuracy**
- Focused on distinguishing Setosa species from the others.
- The model achieved high accuracy, demonstrating logistic regression's capability in binary classification tasks.

**2. Multiclass Classification**
- Implemented a one-vs-all approach for classifying all three iris species.
- The experiments included training separate classifiers for each species and then combining their predictions.
- The results showed satisfactory accuracy levels, indicating the model's effectiveness in multiclass scenarios as well.

**3. Learning Rate Variation**
- Explored the impact of different learning rates (0.001, 0.1, 0.5, 1, 10) on the model's training and convergence.
- Lower learning rates showed slower convergence but more stable performance, while higher rates led to faster but less stable learning, with the potential for divergence.
- The optimal learning rate was found to be around 0.1, balancing speed and stability.

**4. Activation Function Experimentation**
- Compared the performance of different activation functions (ReLU, ELU, Tanh, Sigmoid) in the network.
- ReLU and ELU generally provided better performance due to their non-saturating nature, helping with faster convergence.
- Tanh and Sigmoid resulted in slower convergence and slightly lower accuracy, attributed to their gradients' propensity to vanish.

**5. Dropout Regularization**
- Incorporated dropout in the second fully-connected layer to prevent overfitting, with a rate of 0.3.
- The addition of dropout slightly reduced overfitting, leading to improved test set performance.

**6. The XOR Problem**
- The project concluded with an attempt to model the XOR problem, a non-linear problem logistic regression inherently struggles with due to its linear decision boundary.
- This experiment underscored logistic regression's limitations in handling non-linearly separable data.
