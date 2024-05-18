# Machine Learning playground, tools and tutorials!!!


## Part 1, machine learning playground and object detection tools in streamlit!
This part contains demos of my streamlit website.

### Credit card fraud detection demo
https://github.com/MikheilKvizhinadze2001/ML_playground/assets/85734592/66ddbca1-15bf-4d71-a391-be1ab9ac9f40




**shap values:**



![shap_values](https://github.com/MikheilKvizhinadze2001/ML_playground/assets/85734592/097980bc-fe6e-4118-b6f5-7aa84ee06322)






### Object detection and computer vision demo


https://github.com/MikheilKvizhinadze2001/ML_playground/assets/85734592/942310ab-529a-4dc0-a82c-53cefebb7dfc



### Playground demo


https://github.com/MikheilKvizhinadze2001/ML_playground/assets/85734592/ba646a80-84ab-4074-a1f6-fc42f4ea6cc9



## Part 2, tutorials
Directory called 'tutorials' contains several jupyter notebook files, feel free to check them out for more info, thanks!

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

If you have any questions, contact me at mikheilkvizhinadze@gmail.com