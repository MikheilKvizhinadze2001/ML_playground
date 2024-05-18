import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,make_scorer, recall_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
import joblib
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from scipy.stats import randint, uniform
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import os
import gdown


@st.cache_data
def train_model_with_random_search(X_train, y_train):
    param_space = {
        'n_estimators': Integer(50, 200),
        'max_depth': Integer(3, 8),
        'learning_rate': Real(0.01, 0.1, prior='uniform'),
        'subsample': Real(0.6, 0.9, prior='uniform'),
        'colsample_bytree': Real(0.6, 0.9, prior='uniform'),
        'min_child_weight': Integer(1, 6),
        'gamma': Real(0, 0.5, prior='uniform'),
        'reg_lambda': Real(1, 10, prior='uniform'),
        'reg_alpha': Real(0, 5, prior='uniform')
    }

    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss'
    )
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    recall_scorer = make_scorer(recall_score, average='binary')

    bayes_search = BayesSearchCV(
        estimator=xgb, 
        search_spaces=param_space, 
        n_iter=30,  # Number of iterations for optimization
        cv=skf, 
        n_jobs=-1, 
        scoring=recall_scorer,  # Using recall as the scoring metric
        verbose=1,
        random_state=42
    )

    bayes_search.fit(X_train, y_train)

    best_params = bayes_search.best_params_
    st.write(f"Best parameters: {best_params}")

    return bayes_search.best_estimator_, best_params

@st.cache_data
def train_model_tuned(X_train, y_train, best_params):
    # Add fixed parameters to best_params to avoid conflicts
    best_params.update({
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    })
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              early_stopping_rounds=10,
              verbose=False)
    return model

# Function to evaluate the model
@st.cache_data
def evaluate_model(_model, X_test, y_test):
    # Make predictions on the test set
    y_pred = _model.predict(X_test)

    # Display the classification report
    st.subheader('Classification Report')
    st.write(classification_report(y_test, y_pred))

# Function to compute SHAP valuess
@st.cache_data
def compute_shap_values(_model, X):
    explainer = shap.Explainer(_model)
    shap_values = explainer.shap_values(X)
    return shap_values



@st.cache_data
def balance_data_adasyn(X_train, y_train):
    adasyn = ADASYN(random_state=42)
    X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

@st.cache_data
def load_data(url):
    # Create a temporary file
    temp_file = 'temp.csv'

    # Download the file from Google Drive
    gdown.download(url, temp_file, quiet=False)

    # Load the data into a DataFrame
    df = pd.read_csv(temp_file)

    # Delete the temporary file
    os.remove(temp_file)

    return df

@st.cache_data
@st.experimental_fragment
def show_rows(df, num_rows):
    if choice < 1 or num_rows > len(df):
        st.error("Invalid input! Please select a value between 1 and {}".format(len(df)))
    else:
        # proceed with the selected number of rows
        st.write(f'Here are the first {num_rows} rows of the fraud dataset:')
        st.write(df.head(num_rows))

@st.experimental_fragment
def plot_pie_chart(df):
    # Count the occurrences of fraud and no fraud cases
    fraud_counts = df['Class'].value_counts()

    # Create a pie chart
    fig = px.pie(values=fraud_counts, names=fraud_counts.index, title='Class Distribution')

    # Display the pie chart
    st.plotly_chart(fig)

@st.cache_data

def plot_transactions(df):
    # Transform the 'Time' feature into hours
    df['hour'] = df['Time'] / 3600 % 24

    # Separate fraud and normal transactions
    fraud = df[df['Class'] == 1]
    normal = df[df['Class'] == 0]

    # Create a histogram for normal transactions
    trace0 = go.Histogram(
        x=normal['hour'],
        opacity=0.75,
        name='Normal',
        marker=dict(color='rgba(12, 50, 196, 0.6)'))

    # Create a histogram for fraudulent transactions
    trace1 = go.Histogram(
        x=fraud['hour'],
        opacity=0.75,
        name='Fraud',
        marker=dict(color='rgba(246, 78, 139, 0.6)'))

    # Add the two histograms to the data list
    data = [trace0, trace1]

    layout = go.Layout(barmode='overlay',
                       title='Transactions by Hour',
                       xaxis=dict(title='Time (in Hours)'),
                       yaxis=dict( title='Count'),
                       )

    fig = go.Figure(data=data, layout=layout)

    # Display the plot
    st.plotly_chart(fig)

@st.cache_data
def plot_correlation_matrix(df):
    # Calculate the correlation matrix
    corr = df.corr()

    # Create a heatmap
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).values,
        showscale=True)

    # Update layout
    fig.update_layout(title='Correlation Matrix',
                      width=800, height=800)

    # Display the plot
    st.plotly_chart(fig)

@st.cache_data
def split_dataframe(df, test_size=0.2):
    # Split the data into features and target variable
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Encode the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify = y_encoded, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test



@st.cache_data
def train_model(X_train, y_train):
    # Initialize the model
    model = XGBClassifier(scale_pos_weight=1, use_label_encoder=False)

    # Train the model
    model.fit(X_train, y_train)

    return model

@st.cache_data
def get_classification_report(y_test, y_pred):
    # Generate the classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    return report_df



@st.cache_data
def plot_shap_values(shap_values, X):
    # Initialize the plot
    shap.summary_plot(shap_values, X, plot_type="bar")



# Title
st.title("Credit Card Fraud Detection")

# Introduction
st.subheader("Introduction")
st.write("""
Welcome to this interactive application which showcases a machine learning model trained to detect fraudulent credit card transactions. This project is a demonstration of my data science skills, including data preprocessing, model building, model evaluation, and app development.
""")

# Dataset
st.subheader("Dataset")
st.write("""
The dataset used in this project is the Credit Card Fraud Detection dataset from Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
It contains only numerical input variables which are the result of a PCA transformation. Due to confidentiality issues, original features are hidden. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
""")

# Methodology
st.subheader("Methodology")
st.write("""
The project follows these steps:
1. **Data Preprocessing**: The data is cleaned and preprocessed for the model.
2. **Exploratory Data Analysis (EDA)**: The data is analyzed to understand the distribution of variables and the relationship between different features.
3. **Model Building**: A suitable machine learning model is trained on the preprocessed data.
4. **Model Evaluation**: The performance of the model is evaluated using appropriate metrics.
5. **Model Interpretation**: Techniques like SHAP are used to understand the contribution of each feature to the prediction.
""")

# About the App
st.subheader("About the App")
st.write("""
This app allows you to explore the dataset, understand the model's performance, and even make your own predictions. Navigate through the different sections to learn more about the project and see the model in action.
""")

# Call to Action
st.subheader("Explore")
st.write("""
Feel free to explore the app and don't hesitate to reach out if you have any questions or feedback. Enjoy!
""")


url = 'https://drive.google.com/uc?export=download&id=1TynhlIbI1BI2Ug2bB7mSooYCpiXEN3Vk'
df = load_data(url)
st.write("Now let us inspect the dataset a bit 👇")

st.write("Summary of the dataset:")
st.write(df.describe())

choice = st.number_input(f'**Next**: As we see, dataset has {len(df)} rows and fortunately, there are no missing values. How many rows would you like to see?', min_value=1, max_value=len(df))

show_rows(df, choice)

st.write("Now let's take a look at the distribution of the target variable 'Class', or in other words, the distribution of fraud and no fraud cases in the dataset.")

plot_pie_chart(df)

st.write("As we see, there are only 492 fraud cases out of 284,807 transactions, which makes the dataset highly imbalanced. This imbalance can make it challenging for the model to detect fraud cases accurately.")

st.write("Next, let's visualize the distribution of transactions over time to see if there are any patterns in the data.")
plot_transactions(df)


st.subheader("Feature Analysis")
st.write("Now, let's analyze the features in the dataset to see if there are any differences between normal and fraudulent transactions.")
st.write("Let us plot the correlation matrix to see the relationships between different features.")

# Plotting the correlation matrix
plot_correlation_matrix(df)

st.write("""
        As we see from the correlation matrix, most of the features are not highly correlated with each other, which is good for the model. However, the features V17, V14, V12, and V10 have a relatively higher correlation with the target variable 'Class', indicating that they might be important for detecting fraud cases.
        However, the correlation matrix only shows linear relationships between features. Correlation is a measure of linear relationships and might not capture non-linear relationships or accurately represent the relationship between features and the target variable in imbalanced datasets
        To understand the non-linear relationships and feature importances, we can use techniques like SHAP (SHapley Additive exPlanations).
        SHAP is a powerful technique that explains the output of any machine learning model by calculating the contribution of each feature to the prediction. Let's use SHAP to interpret the model and understand the importance of each feature in detecting fraud cases.
         """)

st.write("But before we use SHAP, we need to train a machine learning model on the dataset. Let's proceed with model building and evaluation.")


st.subheader("Model Building")

st.write("""
         Let us split the dataset into training and testing sets. We will use 80% of the data for training and 20% for testing. This split will help us evaluate the model's performance on unseen data.
         But, because the dataset is highly imbalanced, we will use stratified sampling to ensure that the proportion of fraud and no fraud cases is the same in both the training and testing sets.
         """)

X_train, X_test, y_train, y_test = split_dataframe(df)
st.write("""
         So, what is the class distribution in the training set?
         """)
st.write("Number of fraud cases in the **training set:** {}".format(sum(y_train == 1)))
st.write("And this is the class distribution in the **testing set:**")
st.write("Number of fraud cases in the testing set: {}".format(sum(y_test == 1)))
st.write("So, did the percentage of fraud cases remain the same in the training and testing sets?")
st.write("Percentage of fraud cases in the training set: (100 * sum(y_train == 1) / len(y_train) = {:.2f}%".format(100 * sum(y_train == 1) / len(y_train)))
st.write("Percentage of fraud cases in the testing set: (100 * sum(y_test == 1) / len(y_test)) =  {:.2f}%".format(100 * sum(y_test == 1) / len(y_test)))



st.subheader("""
         Balancing data using ADASYN
         """)
st.write("""
            To address the class imbalance in the dataset, we will use the Adaptive Synthetic Sampling (ADASYN) algorithm to oversample the minority class (fraud cases) and balance the training data. ADASYN generates synthetic samples for the minority class by interpolating between existing samples based on their density distribution.
            
         """)


# Balance the training data
X_train_res, y_train_res = balance_data_adasyn(X_train, y_train)

X_train_df = pd.DataFrame(X_train, columns=X_train.columns)
y_train_df = pd.Series(y_train, name='Class')

st.header("Model Evaluation (XGBoost Classifier)")
st.subheader("Balanced Data")
st.write("Number of fraud cases before balancing: {}".format(sum(y_train == 1)))
st.write("Number of no fraud cases in the balanced training data: {}".format(sum(y_train_res == 0)))


# we use fragments in conjunction with buttons, because we want to control when the content is displayed
# and we don't want to display it immediately when the page is loaded, nor we want our webpage to reload when the button is clicked

st.write("We will first train the model using the balanced data and then evaluate its performance on the training set.")
# Balanced Data
model = train_model(X_train_res, y_train_res)
# Make predictions on the test set
y_pred = model.predict(X_train)
# Display the classification report
st.subheader('Classification Report of train data (using Balanced Data)')
st.write(get_classification_report(y_train, y_pred))
st.write("As we see, model overfits the data when using default hyperparameters. We can further improve the model's performance by tuning the hyperparameters of the XGBoost classifier.")




# Call the function to train the model
# Non-Balanced Data
st.write("Let us see how the model performs on the test set using the non-balanced data.")
model = train_model(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Display the classification report of non-balanced data
st.subheader('Classification Report (Non-Balanced Data)')
st.write(get_classification_report(y_test, y_pred))



# Balanced Data
st.write("Let us see how the model performs on the test set using the balanced data.")
model = train_model(X_train_res, y_train_res)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Display the classification report
st.subheader('Classification Report (Balanced Data)')
st.write(get_classification_report(y_test, y_pred))




st.header("Improving Performance")
st.write("Our recall score after balancing the data is 0.85, which is a significant improvement over the non-balanced data. However, we can further improve the model's performance by tuning the hyperparameters of the XGBoost classifier.")
st.write("We can use Random Search to find the best hyperparameters for the model. Let's proceed with hyperparameter tuning to improve the model's performance.")
st.write("To save time, I will just display the best hyperparameters and save the best model. You can further improve the model by tuning the hyperparameters and training the model again.")


# Snippet to train model with hyperparameter tuning
# Uncomment the code below to train the model with hyperparameter tuning

# Perform hyperparameter tuning
#best_model, best_params = train_model_with_random_search(X_train, y_train)

# Train model with best parameters
#model = train_model_tuned(X_train, y_train, best_params)

# Save the model
#joblib.dump(model, 'best_model.pkl')

    # Perform hyperparameter tuning
#best_model, best_params = train_model_with_random_search(X_train_res, y_train_res)

    # Train model with best parameters
#model = train_model_tuned(X_train_res, y_train_res, best_params)

# Save the model
#oblib.dump(model, 'best_model.pkl')
# Load best model
best_model = joblib.load('best_xgb_for_fraud.pkl')
# Evaluate the model
#st.write("Model performance on the test set using the best hyperparameters:")
#evaluate_model(model, X_test, y_test)
y_pred = best_model.predict(X_test)
st.write("Model performance on the test set using the best hyperparameters:")
st.write(get_classification_report(y_test, y_pred))
# load best model
#best_model = joblib.load('best_model.pkl')
# Functions to evaluate the model and plot SHAP values

st.write("As we see, the model's performance (recall) has improved significantly after hyperparameter tuning. The recall score is now 0.88, which means that the model can detect 88% of fraud cases in the dataset, compared to 0.8469, which was the recall score before hyperparameter tuning. This improvement in performance is crucial for detecting fraud cases accurately.")
st.write("However, the model's precision has decreased slightly after hyperparameter tuning. This trade-off between precision and recall is common in imbalanced datasets, where the model tries to maximize recall at the expense of precision. Depending on the business requirements, we can adjust the model's threshold to optimize precision or recall.")
st.write("It should be noted that current recall score not perfect, especially for tasks like fraud detection and there is still room for improvement. We can further improve the model's performance by using more advanced techniques like anomaly detection algorithms or ensemble methods. However, for the purpose of this project, the XGBoost classifier with hyperparameter tuning has provided a good balance between precision and recall.")


# SHAP Values 
st.header("Model Interpretation using SHAP")
st.write("Now that we have trained the model and evaluated its performance, let's use SHAP to interpret the model and understand the importance of each feature in detecting fraud cases.")
st.write("SHAP (SHapley Additive exPlanations) is a powerful technique that explains the output of any machine learning model by calculating the contribution of each feature to the prediction.")


shap_values = compute_shap_values(best_model, X_test)
st_shap(shap.summary_plot(shap_values, X_test))

st.write("Thank you for exploring the app! I hope you enjoyed the journey through data preprocessing, model building, evaluation, and interpretation. Feel free to reach out if you have any questions or feedback. Have a great day!")

