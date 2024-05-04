import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# Helper functions for the app


# Function to plot the correlation matrix
def plot_correlation_matrix(df):
    st.write("Now, let us visualize the correlation matrix:")
    corr = df.corr().round(2)

    # Convert DataFrame to list of lists, which is what Plotly's create_annotated_heatmap function expects
    z = corr.values.tolist()

    # Create x and y labels
    labels = corr.columns.tolist()

    # Create annotated heatmap
    fig = ff.create_annotated_heatmap(z, x=labels, y=labels, colorscale='Viridis')

    # Add title
    fig.update_layout(title_text='Correlation Matrix (rounded to 2 decimal places)',
                  # Add xaxis, yaxis title
                  xaxis = dict(title_text = 'Features'),
                  yaxis = dict(title_text = 'Features'))

    # Show figure
    st.plotly_chart(fig)

# Function to train the model
def train_model(model, X_train, X_test, y_train, y_test):
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return y_pred

# Function to scale the data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled



def split_data(df, train_size):

    #if (df )
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    return X_train, X_test, y_train, y_test


st.title('Machine Learning playground 😊')
st.write("""
         This is a playground which will allow you to experiment with various machine learning
         algorithms and datasets. It allows you to select a dataset, visualize and investigate it, tweak parameters (and hyperparameters) and see the 
        results in real-time. This is a great way to learn how machine learning algorithms work and how they can be
        applied to different datasets.
         """)


st.header('Datasets:')
st.subheader('Iris Dataset:')
st.write("""
            The Iris dataset is a classic dataset in machine learning. It contains 150 samples of iris flowers, each with
            4 features: sepal length, sepal width, petal length and petal width. The goal is to classify the flowers into
            one of three species: setosa, versicolor and virginica.
         """)

st.subheader('Wine Dataset:')
st.write(""" 
            The Wine dataset is another classic dataset in machine learning. It contains 178 samples of wine, each with 
            13 features. The goal is to classify the wines into one of three classes: class_0, class_1 and class_2.
            """)
st.subheader('Diabetes Dataset:')
st.write("""
            The Diabetes dataset is a classic dataset in machine learning. It contains 442 samples of diabetes patients,
            each with 10 features. The goal is to predict the progression of diabetes in the patients.
            Target variable is a quantitative measure of disease progression one year after baseline (integer number between 25 and 346)
         
            """)

chosen_dataset = st.selectbox(
    'So, which dataset should we play with ?',
    ('Iris 🪻', 'Wine 🍷', 'Diabetes 🩸'))

if chosen_dataset == 'Iris 🪻':
    dataset = load_iris()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    df['Species'] = df['target'].apply(lambda x: dataset.target_names[x] if 'target_names' in dataset else x)


elif chosen_dataset == 'Wine 🍷':
    dataset = load_wine()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    df['Which class?'] = df['target'].apply(lambda x: dataset.target_names[x] if 'target_names' in dataset else x)


else :
    dataset = load_diabetes()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target


st.write("Now let us inspect the dataset a bit 👇")

st.write("Summary of the dataset:")
st.write(df.describe())

choice = st.number_input(f'So, dataset has {len(df)} rows. How many rows would you like to see?', min_value=1, max_value=len(df))

if choice < 1 or choice > len(df):
    st.error("Invalid input! Please select a value between 1 and {}".format(len(df)))
else:
    # proceed with the selected number of rows
    st.write(f'Here are the first {choice} rows of the {chosen_dataset} dataset:')
    st.write(df.head(choice))


# Visualize the dataset
if chosen_dataset == 'Wine 🍷' or chosen_dataset == 'Iris 🪻':
    st.write(f'We are trying to clasify the {chosen_dataset} dataset into different classes. Let us visualize correlation matrix, which shows the relationships between different features.')
    # Select all columns except the last two
    # Create a new DataFrame with all columns except the last one
    df_without_target_col = df.iloc[:, :-1].copy()

    # Now cols_to_plot is a list of column names, which can be used to index df

    plot_correlation_matrix(df_without_target_col)

else:
    st.write(f"We are trying to predict the progression of diabetes in the patients. Let us visualize correlation matrix, which shows the relationships between different features and output. Many plots, but don't worry 😅")
    for column in df.columns:
        if column != 'target':
            fig = px.scatter(df, x=column, y='target')
            st.plotly_chart(fig)
    st.write("Now, let us visualize the correlation matrix:")
    
    # plot correlation matrix
    plot_correlation_matrix(df)


st.header('Machine Learning Models:')

st.write("""
        Now that we have visualized the dataset, let us try to build some machine learning models on it. We will
        split the dataset into training and testing sets, normalize the features and then train the model on the
         """)

st.write("So, how should we split the dataset into training and testing sets?")
train_size = st.slider('Select the train size (%):', min_value=1, max_value=100, value=80, step=1)
show_hint = st.checkbox('Not sure? Unlock hint! 🤔')
if show_hint:
    st.info("The train size is the percentage of the dataset that will be used for training the model. The remaining percentage will be used for testing the model. A common split is 80% training and 20% testing for small datasets (all of our datasets are small)")

if (train_size == 100):
    st.error("Whoops! No data left for testing 😬. Please select a smaller train size.")

# Save the train size in session state
if ("train_size" not in st.session_state):
    st.session_state.train_size = train_size



if chosen_dataset == 'Diabetes 🩸':
    model = st.selectbox('Select the model you want to train:', ('Random Forest Regressor', 'LinearRegression'))
    st.write("If not sure, don't worry! Just check out all the models and see which one works best for you 😊")
else:
    model = st.selectbox('Select the model you want to train:', ('Decision Tree Classifier', 'Random Forest Classifier', 'Logistic Regression Classifier','K Nearest Neighbors Classifier'))
    st.write("If not sure, don't worry! Just check out all the models and see which one works best for you 😊")




st.image('photos/tree.png',caption='Decision Tree Classifier')
st.image('photos/forest.png',caption='Random Forest Classifier')
st.image('photos/regression.png',caption='Logistic Regression Classifier')
st.image('photos/forest_regress.png',caption='Random Forest Regressor')
st.image('photos/log_regress.jpeg',caption='Logistic Regression (classifier)')



# K Nearest Neighbors Classifier

if model == 'K Nearest Neighbors Classifier':
    
    st.write("""
             KNN is a simple and effective classification algorithm that works by finding the K most similar data points (nearest neighbors) to a new input data point. The algorithm then uses the majority vote of these K nearest neighbors to classify the new data point. In other words, KNN looks at the K most similar data points and asks them to vote on the class label of the new data point. The class label with the most votes is then assigned to the new data point. This process is repeated for each new data point, allowing KNN to make predictions based on the similarity of the data points. The key idea behind KNN is that data points that are close together are likely to have similar class labels, making it a powerful algorithm for classification tasks.
                """)
    st.image('photos/knn.png',caption='K-Nearest Neighbors Classifier')
    n_neighbers = st.number_input('Select the number of neighbors:', min_value=1, max_value=len(df), value=5,)
    if ('model' not in st.session_state):
        st.session_state.model = KNeighborsClassifier(n_neighbors=n_neighbers)
    st.write("Should we normalize the data before training the model? (Scale numerical data to a common range, usually between 0 and 1, to prevent differences in scales from affecting analysis or modeling results.)")
    normalize = st.checkbox('Normalize data')

    X_train, X_test, y_train, y_test = split_data(df_without_target_col, train_size)

    try:
        if (normalize):    
            X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
            
            y_pred = train_model(st.session_state.model, X_train_scaled, X_test_scaled, y_train, y_test)
        else:
            y_pred = train_model(st.session_state.model, X_train, X_test, y_train, y_test)
        
        st.write("Model trained successfully! Here are the predictions and the actual values:")
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        results.index.name = 'index'
        st.write(results)
    except Exception as e:
        st.write(f"An error occurred: {e}")

        st.write("Now, let us evaluate the model using some metrics:")
        
# Decision Tree Classifier
elif model == 'Decision Tree Classifier':
    pass

# Random Forest Classifier
elif model == 'Random Forest Classifier':
    pass
# Logistic Regression Classifier
elif model == 'Logistic Regression Classifier':
    pass