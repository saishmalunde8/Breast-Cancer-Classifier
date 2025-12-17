# Breast Cancer Classifier
* A program to classify breast cancer cells to either malignant or benign. <br />
* The program uses a [Deep Neural Network](https://en.wikipedia.org/wiki/Deep_learning) to classify the cells. <br />
* It gives the user insights into how a DNN model is trained and finally test it on a feature. <br />
* The dataset is loaded from sklearn.datasets

## Program flow
The program takes the user through the whole process starting from loading the dataset to predicting a feature.
* User needs to first load the dataset using the respective button.
* After loading the data, user can see different type of analysis done on the data.
* After analysing the user needs to preprocess the data into test, validation and training data.
* The program consists of customizable DNN network with customizable hyperparameters, thus user can train the model with different number of input and hidden nodes, and can
also train the network with different hyperparameters.
* The program preloads the test and validation dataset generated during data preprocessing and evaluates the accuracy of the model on both of them.
* Finally the user can test the model on a randomly selected feature (which is also randomly choosen by the program and is stored as "feature.csv" in "Datasets" folder).
* The generated test, validation and training dataset are also stored in "Datasets" folder.

## How to run?
* Run the following commands
``` 
pip install -r requirements.txt
py gui.py
```

## Images
### Main UI
![CancerClassifierMenu](https://user-images.githubusercontent.com/55596801/142252628-5114b61e-4206-41e5-81b1-32d3e723c807.png)
### Data Analysis UI
![DataAnalysis](https://user-images.githubusercontent.com/55596801/142252647-b995f8f9-a0a7-4cfd-bb16-6d8e3be49514.png)
### Train Model UI
![TrainModel](https://user-images.githubusercontent.com/55596801/142252665-d061f2ee-244d-4e39-aa22-94892b34e7a2.png)
### Evaluate Model UI
![EvaluateModel](https://user-images.githubusercontent.com/55596801/142253346-c737e715-6a5a-4877-9d89-b8482ad99ab8.png)
### Predict Model UI
![predictModel](https://user-images.githubusercontent.com/55596801/142252692-33bf800c-e5cb-4767-8070-cbdb4dbd7fe6.png)
