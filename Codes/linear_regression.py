#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 21:11:43 2018

@author: abinaya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso

df_train = pd.read_csv("/Users/abinaya/USC/Studies/NLCI/Project/Data/train.csv")
df_test = pd.read_csv("/Users/abinaya/USC/Studies/NLCI/Project/Data/test.csv")

del df_train["Unnamed: 0"]
del df_test["Unnamed: 0"]

df_train_y = df_train["price"]
df_train_x = df_train.drop("price", axis=1)

df_test_y = df_test["price"]
df_test_x = df_test.drop("price", axis=1)

### Linear Regression
lr = Lasso()
lr.fit(df_train_x, df_train_y)

train_acc_LinReg = lr.score(df_train_x, df_train_y)
test_acc_LinReg = lr.score(df_test_x, df_test_y)

print("Linear Regression - Train accuracy: ",train_acc_LinReg )
print("Linear Regression - Test accuracy: ",test_acc_LinReg )

### Logistic Regression
logr = LogisticRegression(C=1)
logr.fit(df_train_x, df_train_y)

train_acc_LogReg = logr.score(df_train_x, df_train_y)
test_acc_LogReg = logr.score(df_test_x, df_test_y)

print("Logistic Regression - Train accuracy: ",train_acc_LogReg )
print("Logistic Regression - Test accuracy: ",test_acc_LogReg )