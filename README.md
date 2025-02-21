# Customer Churn Prediction App

This project demonstrates a machine learning pipeline for predicting customer churn using a web application built with Streamlit. The repository includes code, a dataset, and a detailed notebook showcasing the approach and models used.

## Project Structure

```
├── app.py               # Streamlit app for interactive predictions
├── Dataset.csv          # Dataset used for training and testing models
├── notebook.ipynb       # Jupyter notebook with detailed EDA and ML implementation
├── README.md            # Project overview and documentation
```

## Features

- **Interactive Web Application**: Allows users to select models, view their performance, and analyze confusion matrices interactively.
- **Machine Learning Models**: Implements and evaluates Logistic Regression, Random Forest, and K-Nearest Neighbors.
- **EDA and Visualization**: Detailed exploratory data analysis and visualizations are available in the Jupyter notebook.

## Getting Started

### Prerequisites

Make sure you have Python installed along with the following libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `streamlit`

You can install these dependencies using:

```bash
pip install -r requirements.txt
```

### How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd customer-churn-prediction
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open the app in your browser using the URL displayed in the terminal.

## Dataset

The dataset (`Dataset.csv`) contains customer information along with a target column `Exited` indicating whether a customer churned (1) or not (0).

### Key Columns:

- **CreditScore**: Credit score of the customer.
- **Age**: Age of the customer.
- **Balance**: Customer's account balance.
- **NumOfProducts**: Number of products the customer is subscribed to.
- **IsActiveMember**: Indicates if the customer is active.
- **EstimatedSalary**: Annual estimated salary.
- **Exited**: Target variable (1 = churned, 0 = retained).

## Machine Learning Models

### Logistic Regression
A statistical model used for binary classification. It predicts probabilities for each class and assigns the label based on a threshold (e.g., 0.5).

### Random Forest
An ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting.

### K-Nearest Neighbors (KNN)
A non-parametric algorithm that assigns a label based on the majority class among its nearest neighbors.

## Metrics and Evaluation

### Confusion Matrix
A matrix used to evaluate the performance of a classification model by showing the counts of:
- **True Positives (TP)**: Correctly predicted churned customers.
- **True Negatives (TN)**: Correctly predicted retained customers.
- **False Positives (FP)**: Incorrectly predicted churned customers.
- **False Negatives (FN)**: Incorrectly predicted retained customers.

### Accuracy
Proportion of correctly predicted samples:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Classification Report
Includes precision, recall, and F1-score for each class:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall.

## Results

The app outputs the following for each selected model:
- **Accuracy**: The overall performance of the model.
- **Classification Report**: Detailed metrics for each class.
- **Confusion Matrix**: Visual representation of model performance.

## Future Improvements

- Add more advanced models such as XGBoost or Gradient Boosting.
- Implement hyperparameter tuning for improved performance.
- Allow users to upload custom datasets.

---

Feel free to contribute to this project by submitting issues or pull requests. For questions, reach out via [parwaazjoshi@gmail.com].

