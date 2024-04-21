# Parameter optimization for SVM
## Prisha Sawhney
## 102116052
## 3CS10

## Overview

The program performs Parameter Optimization for SVM.

## Requirements

To run this program, you need:
- Python 3.x installed on your system
- Basic understanding of SVM in Python

## Usage

1. Clone this repository to your local machine
2. Navigate to the repository directory
3. Run the program using any notebook editor

## Methodology
- Import necessary libraries
```
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.preprocessing import StandardScaler
```



- Fetch the dataset from UCI Library
```
# fetch dataset 
dry_bean = fetch_ucirepo(id=602) 
  
# data (as pandas dataframes) 
X = dry_bean.data.features 
y = dry_bean.data.targets
```



- Preprocess the dataset as per requirements
```
df = pd.concat([X, y], axis=1)
df.columns = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Class']
# label encoding the target variable
le = LabelEncoder()
df['Class'] = le.fit_transform(y)
```



- Create 10 samples of the dataset
```
samples = []
for i in range(10):
    samples.append(df.sample(n=2000))
```



- Find the best parameters for SVM
```
dataframes = []
result = pd.DataFrame(columns = ['Sample Number', 'Best Kernel', 'Best C', 'Best Gamma', 'Best Accuracy'])
for num,sample in enumerate(samples, start=1):
    df = pd.DataFrame(columns = ['iteration', 'kernel', 'C', 'gamma', 'accuracy'])
    X = sample.drop(['Class'], axis=1)
    y = sample['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    kernelList = ['linear', 'poly', 'rbf', 'sigmoid']
    best_accuracy = 0
    for i in range(100):
        k = random.random()
        n = random.random()
        kernel = random.choice(kernelList)
        clf = svm.SVC(kernel=kernel, verbose=False, C=k, gamma=n, max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_kernel = kernel
            best_C = k
            best_gamma = n
            df = df.append({'iteration': i, 'kernel': kernel, 'C': k, 'gamma': n, 'accuracy': accuracy}, ignore_index=True)
    dataframes.append(df)
    result = result.append({'Sample Number': "sample {}".format(num), 'Best Kernel': best_kernel, 'Best C': best_C, 'Best Gamma': best_gamma, 'Best Accuracy': best_accuracy}, ignore_index=True)
```



- Analyzing the result of the above code block


![image](https://github.com/prishasawhney/Parameter-Optimization-of-SVM/assets/138293599/1f70c78e-984d-4468-b4af-e1a7a6f8765d)




- Convergence graph of the best Accuracy
![image](https://github.com/prishasawhney/Parameter-Optimization-of-SVM/assets/138293599/7519c0d7-e887-457a-92af-37da2ac56cd3)


As we can see clearly from the graph that the Accuracy increases as the number of iterations is increased


#### Note: The results may vary as the code works on random choices
