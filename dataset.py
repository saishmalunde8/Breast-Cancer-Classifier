import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tkinter as tk

cancer_df = None
X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test = None, None, None, None, None, None

def load_dataset():
    global cancer_df

    cancer_dataset = load_breast_cancer()
    cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'], cancer_dataset['target']], columns = np.append(cancer_dataset['feature_names'], ['target']))

    # tk.messagebox.showinfo("Load Dataset", "Dataset has been loaded.")

    cancer_df.head(6)
    cancer_df.info()

    print("Dataset has been loaded!")

def target_count_plot():
    sns.countplot(x = cancer_df['target'])
    plt.show()

def mean_radius_count_plot():
    plt.figure(figsize = (20, 8))
    sns.countplot(x = cancer_df['mean radius'])
    plt.xticks([])

    plt.show()

def dataset_heatmap():
    plt.figure(figsize = (16, 9))
    sns.heatmap(data = cancer_df)

    plt.show()

def feature_corr_heatmap():
    plt.figure(figsize = (20, 20))
    sns.heatmap(data = cancer_df.corr(), annot = True, cmap = 'coolwarm')

    plt.show()

def target_corr_heatmap():
    cancer_df2 = cancer_df.drop(['target'], axis = 1)
    # print(f"Shape of cancer_df2: {cancer_df2.shape}")

    plt.figure(figsize = (16, 5))
    ax = sns.barplot(x = cancer_df2.corrwith(cancer_df.target).index, y = cancer_df2.corrwith(cancer_df.target))
    ax.tick_params(labelrotation = 90)

    plt.show()

# Prepare train dataset.
def dataset_preprocessing():
    # Global variables
    global cancer_df
    global X_train_sc
    global X_val_sc
    global X_test_sc 
    global y_train 
    global y_val 
    global y_test

    # Removing 'target' column
    X = cancer_df.drop(['target'], axis = 1)

    # Target column
    y = cancer_df['target']

    # Splitting dataset into cross validation and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 1)

    # Normalizing the input datasets
    sc = StandardScaler()

    X_train_sc = np.transpose(sc.fit_transform(X_train))
    X_val_sc = np.transpose(sc.transform(X_val))
    X_test_sc = np.transpose(sc.transform(X_test))

    # print("X train dataset: \n")
    # print(X_train_sc[:5, :])
    # print("\nShape of X train: ", X_train_sc.shape)

    # Saving input train, test and validation dataset locally.
    pd.DataFrame(X_train_sc).to_csv("Datasets/X_train_sc.csv")
    pd.DataFrame(X_val_sc).to_csv("Datasets/X_val_sc.csv")
    pd.DataFrame(X_test_sc).to_csv("Datasets/X_test_sc.csv")

    # Preparing y as inputs for models
    y_train = np.array(y_train)
    y_train = y_train.reshape(1, y_train.shape[0])

    y_test = np.array(y_test)
    y_test = y_test.reshape(1, y_test.shape[0])

    y_val = np.array(y_val)
    y_val = y_val.reshape(1, y_val.shape[0])

    # print("y train dataset: \n")
    # print(y_train[:5, :])
    # print("\nShape of y_train: ", y_train.shape)

    # Saving y train, test and validation dataset locally.
    pd.DataFrame(y_train).to_csv("Datasets/y_train.csv")
    pd.DataFrame(y_val).to_csv("Datasets/y_val.csv")
    pd.DataFrame(y_test).to_csv("Datasets/y_test.csv")

    # Getting a feature.csv file on which the model can predict on.
    feature = X_test_sc[:, 0]
    y_feature_test = y_test[0, 0]
    print("y_feature_test: ", y_feature_test)
    print("Feature shape: ", feature.shape)
    pd.DataFrame(feature).to_csv("Datasets/feature.csv")

    print("Data preprocessing and data analyzing has been done.")

def get_train_dataset():
    return X_train_sc, y_train

def get_val_dataset():
    return X_val_sc, y_val

def get_test_dataset():
    return X_test_sc, y_test

# load_dataset()
# dataset_preprocessing()