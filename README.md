# Machine Learning playground, tools and tutorials!!!


## Part 1, machine learning playground and object detection tools in streamlit!
This part contains demos of my streamlit website.


### To Run the streamlit app on your local machine (Part 1 of this repo), do the following:

clone the repo using:

git clone https://github.com/MikheilKvizhinadze2001/ML_playground.git

Next, make sure you are in the same directory as 'index.py', and install the dependences using:

pip install -r requirements.txt

Finally, run the streamlit app using:

streamlit run index.py


P.S. You can ignore the 'notebooks' directory, it does not influence streamlit in any way :)



### Credit card fraud detection demo
**Dataset**
The dataset used is the Credit Card Fraud Detection dataset from Kaggle. It contains transactions made by European cardholders in September 2013. With 492 frauds out of 284,807 transactions, it presents a highly imbalanced class distribution. Features include transformed numerical variables and 'Time' and 'Amount' as non-transformed variables.

**Methodology**
The project follows these steps:

- Data Preprocessing: Cleaning and preprocessing the data.
- Exploratory Data Analysis (EDA): Analyzing variable distributions and relationships.
- Model Building: Training a suitable machine learning model.
- Model Evaluation: Assessing model performance using appropriate metrics.
- Model Interpretation: Utilizing techniques like SHAP for feature importance.


https://github.com/MikheilKvizhinadze2001/ML_playground/assets/85734592/66ddbca1-15bf-4d71-a391-be1ab9ac9f40




**shap values of xgboost model used in this project**



![shap_values](https://github.com/MikheilKvizhinadze2001/ML_playground/assets/85734592/097980bc-fe6e-4118-b6f5-7aa84ee06322)






### Real-time object detection demo


https://github.com/MikheilKvizhinadze2001/ML_playground/assets/85734592/942310ab-529a-4dc0-a82c-53cefebb7dfc



### Playground demo

This is a playground which will allow you to experiment with various machine learning algorithms and datasets. It allows you to select a dataset, visualize and investigate it, tweak parameters (and hyperparameters) and see the results in real-time. This is a great way to learn how machine learning algorithms work and how they can be applied to different datasets.



https://github.com/MikheilKvizhinadze2001/ML_playground/assets/85734592/ba646a80-84ab-4074-a1f6-fc42f4ea6cc9




### Video annotation tool demo
The tool offers the following functionality:

- Video Upload: Users can upload a video file of their choice.
- Annotation: The uploaded video is annotated with relevant information or markings.
- Output: An annotated version of the video is generated and made available for download.

  
## Video before:



https://github.com/MikheilKvizhinadze2001/ML_playground/assets/85734592/22f0c983-0dd2-46e1-80b2-73c48cfc7dcb



## Video after:



https://github.com/MikheilKvizhinadze2001/ML_playground/assets/85734592/316e86e7-5077-41a1-8db5-1c41d116119e



## Part 2, tutorials
Directory called 'notebooks' contains several jupyter notebook files, feel free to check them out for more info, thanks!

This part contains various machine learning projects, each demonstrating different techniques and algorithms on distinct datasets. The projects include ensemble learning, K-Means clustering, rice image classification using CNNs, and cat vs dog image classification.

**1. Ensemble Learning on MNIST Dataset**
Goal: Classify handwritten digits using ensemble learning techniques.

Steps
Data Preparation: Load and split the MNIST dataset.
Training Individual Classifiers: Train Random Forest, Extra Trees, SVM, and MLP classifiers.
Voting Classifier: Combine classifiers into a voting classifier for majority vote predictions.
Model Evaluation: Assess individual and ensemble model performance on validation and test sets.
Improvement and Blending: Optimize the voting classifier by removing SVM, and create a blender and stacking classifier.
Comparison: Compare the performance of different ensemble methods.
Key Findings
The stacking classifier outperformed other methods.
Removing SVM improved the voting classifier’s performance.
The blender was less effective due to various factors including individual prediction quality.
Conclusion
Ensemble methods enhance model performance, with the best approach varying by dataset and model specifics.

**2. K-Means Clustering on Customer Dataset**
Goal: Implement K-Means clustering to segment customers.

Dataset: Customer Segmentation Dataset from Kaggle.

Implementation
Library Implementation: Benchmark K-Means using libraries.
Scratch Implementation: Manually implement K-Means, including centroid initialization, data point assignment, and centroid updates.
Link to Dataset: 
https://www.kaggle.com/datasets/yasserh/customer-segmentation-dataset

**3. Rice Image Classification using CNNs**
Goal: Classify rice images into five types using a Convolutional Neural Network (CNN).

Dataset: 75,000 rice images, with 15,000 images per class.

Steps
Model Architecture: Three convolutional layers with max-pooling, followed by a fully connected layer and softmax output.
Training: Use Adam optimizer and sparse categorical cross-entropy loss. Apply data augmentation techniques.
Evaluation: Assess model accuracy on a test set and visualize training/validation loss and accuracy.
Link to Dataset: 
https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset


**4. Cat vs Dog Image Classification**
Goal: Classify images of cats and dogs using different models.

Steps
Logistic Regression: Baseline model, inadequate for complex image data.
Random Forest: More complex but prone to overfitting. Requires hyperparameter tuning.
VGG16: Pre-trained CNN model, best performance, but underfits the data. Potential improvements with increased complexity and fine-tuning.
Link to Dataset: 
https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset

### Conclusion
These projects collectively showcase various machine learning techniques applied to different datasets. They highlight the strengths and limitations of each method, emphasizing the importance of model selection and tuning based on specific tasks and datasets.

### Update
As of 10/22/2024, the app is inactive, but of course you can always reproduce the results :)

If you have any questions, contact me at mikheilkvizhinadze@gmail.com
