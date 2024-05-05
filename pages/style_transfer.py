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
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Helper functions for the app


# Function to plot the correlation matrix
@st.cache_data
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
@st.cache_data
def train_model(model, X_train, X_test, y_train):
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return y_pred

# Function to scale the data
@st.cache_data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


@st.cache_data
def evaluate_model(y_pred, y_test):
        st.success("Model trained successfully! Here are the predictions and the actual values:", icon="✅")
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        results.index.name = 'index'
        st.write(results)

        st.subheader("Now, let us evaluate the model using some metrics:")
        st.write("Accuracy Score (percentage of correct predictions):")
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"{accuracy * 100:.2f}%, were correctly classified.")
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
      
        # Display the confusion matrix using seaborn's heatmap
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)

        # Display the classification report
        st.subheader('Classification Report')
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)


@st.cache_data
def evaluate_regression_model(y_pred, y_test):
    st.success("Model trained successfully! Here are the predictions and the actual values:", icon="✅")
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results.index.name = 'index'
    st.write(results)

    st.subheader("Now, let us evaluate the model using some metrics:")
    st.write("Mean Absolute Error (MAE):")
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"{mae:.2f}")

    st.write("Mean Squared Error (MSE):")
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"{mse:.2f}")

    st.write("Root Mean Squared Error (RMSE):")
    rmse = np.sqrt(mse)
    st.write(f"{rmse:.2f}")


# Function to split the data into training and testing sets
@st.cache_data
def split_data(df, train_size):


    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    train_size = train_size / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    return X_train, X_test, y_train, y_test

# Function to perform cross-validation on regression tasks
@st.cache_data
def cross_validation_regression(model, df, cv):
    """
    Perform cross-validation on a given model.

    Parameters:
    model: The scikit-learn model to evaluate.
    df: The dataset to use for cross-validation.
    cv: The number of folds for cross-validation.

    Returns:
    None, because the results are displayed in Streamlit.
    """
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    scores = cross_val_score(model, X, y, cv=int(cv), scoring=scorer)
    scores = np.abs(scores)  # Take the absolute value of the scores
    # Calculate the mean and standard deviation of the scores
    mean_score = np.mean(scores)
    std_dev = np.std(scores)

    # Display the scores in Streamlit
    st.write(f"Cross-validation scores: {scores}")
    st.write(f"Mean score: {mean_score:.2f}")
    st.write(f"Standard deviation: {std_dev:.2f}")
    return None

# Function to perform cross-validation on classification tasks
@st.cache_data
def cross_validation(model, df, cv):
    """
    Perform cross-validation on a given model.

    Parameters:
    model: The scikit-learn model to evaluate.
    df: The dataset to use for cross-validation.
    cv: The number of folds for cross-validation.

    Returns:
    None, because the results are displayed in Streamlit.
    """
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    scores = cross_val_score(model, X, y, cv=int(cv))

    # Calculate the mean and standard deviation of the scores
    mean_score = np.mean(scores)
    std_dev = np.std(scores)

    # Display the scores in Streamlit
    st.write(f"Cross-validation scores: {scores}")
    st.write(f"Mean score: {mean_score:.2f}")
    st.write(f"Standard deviation: {std_dev:.2f}")
    return None



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
        training set. Finally, we will evaluate the model on the testing set and perform cross-validation to get a
        better estimate of the model's performance.
        We will use the following machine learning models for classification tasks:
        - Decision Tree Classifier
        - Random Forest Classifier
        - Logistic Regression Classifier
        - K Nearest Neighbors Classifier
         """)
st.write("""
        And the following machine learning models for regression tasks:
        - Random Forest Regressor
        - Linear Regression
         """)
st.write("So, how should we split the dataset into training and testing sets?")
train_size = st.slider('Select the train size (%):', min_value=1, max_value=100, value=80, step=1)
show_hint = st.checkbox('Not sure? Unlock the hint! 🤔')
if show_hint:
    st.info("The train size is the percentage of the dataset that will be used for training the model. The remaining percentage will be used for testing the model. A common split is 80% training and 20% testing for small datasets (all of our datasets are small)")

if (train_size == 100):
    st.error("Whoops! No data left for testing 😬. Please select a smaller train size.")

# Save the train size in session state
if ("train_size" not in st.session_state):
    st.session_state.train_size = train_size

st.write(f"Great! We will use {train_size}% of the dataset, {int(len(df)*(train_size/100))} for training the model and the remaining {100 - train_size}%, {len(df) - int(len(df)*(train_size/100))} for testing the model.")

if chosen_dataset == 'Diabetes 🩸':
    model = st.selectbox('Select the model you want to train:', ('Random Forest Regressor', 'LinearRegression'))
    st.write("If not sure, don't worry! Just check out all the models and see which one works best for you 😊")
else:
    model = st.selectbox('Select the model you want to train:', ('Decision Tree Classifier', 'Random Forest Classifier', 'Logistic Regression Classifier','K Nearest Neighbors Classifier'))
    st.write("If not sure, don't worry! Just check out all the models and see which one works best for you 😊")



_ = """
st.image('photos/tree.png',caption='Decision Tree Classifier')
st.image('photos/forest.png',caption='Random Forest Classifier')
st.image('photos/regression.png',caption='Logistic Regression Classifier')
st.image('photos/forest_regress.png',caption='Random Forest Regressor')
st.image('photos/log_regress.jpeg',caption='Logistic Regression (classifier)')
"""



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
    st.write(f"train size {len(X_train)} ")
    st.write(f"test size {len(X_test)}")
    try:
        if (normalize):    
            X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
            
            y_pred = train_model(st.session_state.model, X_train_scaled, X_test_scaled, y_train)
        else:
            y_pred = train_model(st.session_state.model, X_train, X_test, y_train)
        
        # Evaluate the model
        if st.button("Shall we evaluate the model?"):
            evaluate_model(y_pred, y_test)
        # Cross-validation
        st.write("Now let us perform cross-validation on the model, which is a technique used to evaluate the performance of a machine learning model. It involves splitting the dataset into k equal parts (or folds), training the model on k-1 folds and testing it on the remaining fold. This process is repeated k times, with each fold serving as the test set exactly once. The final performance metric is the average of the performance metrics obtained from each fold.")
        st.write("So, how many folds should we use for cross-validation?")
        cv_num = st.text_input('Enter the number of folds [2, inf), integers only.', value=5)
  
        # Evaluate the model
        if st.button("Press to start cross-validation"):
            scores = cross_validation(st.session_state.model, df_without_target_col, cv_num)
            if st.button("Shall we clear the data, and start fresh?"):
                st.session_state.model = None
                st.rerun()
    except Exception as e:
        st.write(f"An error occurred: {e}")


# Decision Tree Classifier
elif model == 'Decision Tree Classifier':
    st.write("""
                A decision tree is a flowchart-like structure in which each internal node represents a feature (or attribute), each branch represents a decision rule, and each leaf node represents the outcome. The topmost node in a decision tree is known as the root node. It learns to partition the data into subsets based on the values of the features. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. Decision trees are easy to understand and interpret, making them a popular choice for classification tasks.
                    
             """)
    st.image('photos/tree.png',caption='Decision Tree Classifier')
    st.write("So, how many levels should the decision tree have?")
    st.info("""
            💡 The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            So, risk of overfitting increases with the depth of the tree.
            """)
    max_depth = st.number_input('Select the maximum depth of the tree:', min_value=1, value=5)
    
    if ('model' not in st.session_state):
        st.session_state.model = DecisionTreeClassifier(max_depth=max_depth)
    st.write("Should we normalize the data before training the model? (Scale numerical data to a common range, usually between 0 and 1, to prevent differences in scales from affecting analysis or modeling results.)")
    normalize = st.checkbox('Normalize data')
    X_train, X_test, y_train, y_test = split_data(df_without_target_col, train_size)

    try:
        if (normalize):    
            X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
            
            y_pred = train_model(st.session_state.model, X_train_scaled, X_test_scaled, y_train)
        else:
            y_pred = train_model(st.session_state.model, X_train, X_test, y_train)


        # Evaluate the model
        if st.button("Shall we evaluate the model?"):
            evaluate_model(y_pred, y_test)

        
        # Cross-validation
        st.write("Now let us perform cross-validation on the model, which is a technique used to evaluate the performance of a machine learning model. It involves splitting the dataset into k equal parts (or folds), training the model on k-1 folds and testing it on the remaining fold. This process is repeated k times, with each fold serving as the test set exactly once. The final performance metric is the average of the performance metrics obtained from each fold.")
        st.write("So, how many folds should we use for cross-validation?")
        cv_num = st.text_input('Enter the number of folds [2, inf), integers only.', value=5)

         # Evaluate the model
        if st.button("Press to start cross-validation"):
            scores = cross_validation(st.session_state.model, df_without_target_col, cv_num)
            if st.button("Shall we clear the data, and start fresh?"):
                st.session_state.model = None
                st.rerun()

    except Exception as e:
        st.write(f"An error occurred: {e}")



# Random Forest Classifier
elif model == 'Random Forest Classifier':
    st.write("""
                A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the `max_samples` parameter if `bootstrap=True` (default), otherwise the whole dataset is used to build each tree.
             """)
    st.image('photos/forest.png',caption='Random Forest Classifier')
    st.write("How many estimators (trees) should the random forest have?")
    st.info("""
            💡 The number of trees in the forest. Increasing the number of trees can lead to improved accuracy, but it can also increase the computational complexity and may lead to overfitting.
            """)
    n_estimators = st.number_input('Select the number of estimators:', min_value=1, value=100)
    
    if ('model' not in st.session_state):
        st.session_state.model = RandomForestClassifier(n_estimators=n_estimators)
        st.write("Should we normalize the data before training the model? (Scale numerical data to a common range, usually between 0 and 1, to prevent differences in scales from affecting analysis or modeling results.)")
    normalize = st.checkbox('Normalize data')
    X_train, X_test, y_train, y_test = split_data(df_without_target_col, train_size)

    try:
        if (normalize):    
            X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
            
            y_pred = train_model(st.session_state.model, X_train_scaled, X_test_scaled, y_train)
        else:
            y_pred = train_model(st.session_state.model, X_train, X_test, y_train)


        # Evaluate the model
        if st.button("Shall we evaluate the model?"):
            evaluate_model(y_pred, y_test)

        
        # Cross-validation
        st.write("Now let us perform cross-validation on the model, which is a technique used to evaluate the performance of a machine learning model. It involves splitting the dataset into k equal parts (or folds), training the model on k-1 folds and testing it on the remaining fold. This process is repeated k times, with each fold serving as the test set exactly once. The final performance metric is the average of the performance metrics obtained from each fold.")
        st.write("So, how many folds should we use for cross-validation?")
        cv_num = st.text_input('Enter the number of folds [2, inf), integers only.', value=5)

         # Evaluate the model
        if st.button("Press to start cross-validation"):
            scores = cross_validation(st.session_state.model, df_without_target_col, cv_num)
            if st.button("Shall we clear the data, and start fresh?"):
                st.session_state.model = None
                st.rerun()

    except Exception as e:
        st.write(f"An error occurred: {e}")

# Logistic Regression Classifier
elif model == 'Logistic Regression Classifier':
    st.write("""
                Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression).
                Our model will use One vs Rest (OvR) multiclass strategy, which involves training a single classifier per class, with the samples of that class as positive 
                samples and all other samples as negatives. This strategy requires the base classifiers to produce a real-valued confidence score for its decision, rather than just a class label.
                So, we would have a separate classifier for each class, which would predict the probability of that class being the correct one.
             """)
    st.image('photos/log_regress.jpeg',caption='Logistic Regression Classifier')
    st.write("What should be the inverse of regularization strength for logistic regression?")
    st.info("""
            💡 Inverse of regularization strength is   C = 1/λ, where λ is the regularization parameter. Regularization is a technique used to prevent overfitting by penalizing large coefficients. A smaller value of C means stronger regularization, which can help prevent overfitting. A larger value of C means weaker regularization, which can lead to overfitting.
            """)
    C = st.number_input('Select the inverse of regularization strength:', min_value=0.1, value=1.0)
    
    if ('model' not in st.session_state):
        st.session_state.model = LogisticRegression(C=C)

    st.write("Should we normalize the data before training the model? (Scale numerical data to a common range, usually between 0 and 1, to prevent differences in scales from affecting analysis or modeling results.)")
    normalize = st.checkbox('Normalize data')
    X_train, X_test, y_train, y_test = split_data(df_without_target_col, train_size)

    try:
        if (normalize):    
            X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
            
            y_pred = train_model(st.session_state.model, X_train_scaled, X_test_scaled, y_train)
        else:
            y_pred = train_model(st.session_state.model, X_train, X_test, y_train)


        # Evaluate the model
        if st.button("Shall we evaluate the model?"):
            evaluate_model(y_pred, y_test)

        
        # Cross-validation
        st.write("Now let us perform cross-validation on the model, which is a technique used to evaluate the performance of a machine learning model. It involves splitting the dataset into k equal parts (or folds), training the model on k-1 folds and testing it on the remaining fold. This process is repeated k times, with each fold serving as the test set exactly once. The final performance metric is the average of the performance metrics obtained from each fold.")
        st.write("So, how many folds should we use for cross-validation?")
        cv_num = st.text_input('Enter the number of folds [2, inf), integers only.', value=5)

         # Evaluate the model
        if st.button("Press to start cross-validation"):
            scores = cross_validation(st.session_state.model, df_without_target_col, cv_num)
            if st.button("Shall we clear the data, and start fresh?"):
                st.session_state.model = None
                st.rerun()

    except Exception as e:
        st.write(f"An error occurred: {e}")

# Random Forest Regressor
elif model == 'Random Forest Regressor':
    st.write("""
                A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the `max_samples` parameter if `bootstrap=True` (default), otherwise the whole dataset is used to build each tree.
             """)
    st.image('photos/forest_regress.png',caption='Random Forest Regressor')
    st.write("How many estimators (trees) should the random forest have?")
    st.info("""
            💡 The number of trees in the forest. Increasing the number of trees can lead to improved accuracy, but it can also increase the computational complexity and may lead to overfitting.
            """)
    n_estimators = st.number_input('Select the number of estimators:', min_value=1, value=100)
    
    if ('model' not in st.session_state):
        st.session_state.model = RandomForestRegressor(n_estimators=n_estimators)

    st.write("Should we normalize the data before training the model? (Scale numerical data to a common range, usually between 0 and 1, to prevent differences in scales from affecting analysis or modeling results.)")
    normalize = st.checkbox('Normalize data')
    X_train, X_test, y_train, y_test = split_data(df, train_size)

    try:
        if (normalize):    
            X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
            
            y_pred = train_model(st.session_state.model, X_train_scaled, X_test_scaled, y_train)
        else:
            y_pred = train_model(st.session_state.model, X_train, X_test, y_train)

        # Evaluate the model
        if st.button("Shall we evaluate the model?"):
            evaluate_regression_model(y_pred, y_test)

        
        # Cross-validation
        st.write("Now let us perform cross-validation on the model, which is a technique used to evaluate the performance of a machine learning model. It involves splitting the dataset into k equal parts (or folds), training the model on k-1 folds and testing it on the remaining fold. This process is repeated k times, with each fold serving as the test set exactly once. The final performance metric is the average of the performance metrics obtained from each fold.")
        st.write("So, how many folds should we use for cross-validation?")
        cv_num = st.text_input('Enter the number of folds [2, inf), integers only.', value=5)

        # Evaluate the model
        if st.button("Press to start cross-validation"):
            scores = cross_validation_regression(st.session_state.model, df, cv_num)
            if st.button("Shall we clear the data, and start fresh?"):
                st.session_state.model = None
                st.rerun()

    except Exception as e:
        st.write(f"An error occurred: {e}")


# Linear Regression
elif model == 'LinearRegression':
    st.write("""
                Linear regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables. In linear regression, the relationships are modeled using linear predictor functions whose unknown model parameters are estimated from the data.
             """)
    st.image('photos/regression.png',caption='Linear Regression')
    
    if ('model' not in st.session_state):
        st.session_state.model = LinearRegression()

    st.write("Should we normalize the data before training the model? (Scale numerical data to a common range, usually between 0 and 1, to prevent differences in scales from affecting analysis or modeling results.)")
    normalize = st.checkbox('Normalize data')
    X_train, X_test, y_train, y_test = split_data(df, train_size)

    try:
        if (normalize):    
            X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
            
            y_pred = train_model(st.session_state.model, X_train_scaled, X_test_scaled, y_train)
        else:
            y_pred = train_model(st.session_state.model, X_train, X_test, y_train)

        # Evaluate the model
        if st.button("Shall we evaluate the model?"):
            evaluate_regression_model(y_pred, y_test)

        
        # Cross-validation
        st.write("Now let us perform cross-validation on the model, which is a technique used to evaluate the performance of a machine learning model. It involves splitting the dataset into k equal parts (or folds), training the model on k-1 folds and testing it on the remaining fold. This process is repeated k times, with each fold serving as the test set exactly once. The final performance metric is the average of the performance metrics obtained from each fold.")
        st.write("So, how many folds should we use for cross-validation?")
        cv_num = st.text_input('Enter the number of folds [2, inf), integers only.', value=5)

        # Evaluate the model
        if st.button("Press to start cross-validation"):
            scores = cross_validation_regression(st.session_state.model, df, cv_num)
            if st.button("Shall we clear the data, and start fresh?"):
                st.session_state.model = None
                st.rerun()

        if st.button("Shall we clear the data, and start fresh?"):
            st.session_state.model = None
            st.rerun()
    except Exception as e:
        st.write(f"An error occurred: {e}")