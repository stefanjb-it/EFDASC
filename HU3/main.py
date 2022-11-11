from collections import Counter

import pandas as pd
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


# np.set_printoptions(threshold=sys.maxsize)

def BSP_1():
    df = pd.read_csv('./data/adult.csv')

    df = df.drop(df.columns[2], axis=1)
    df.columns = ['age', 'workclass', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
                  'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

    df = df[(df['hours_per_week'] >= 30) & (df['hours_per_week'] <= 50)]

    df = df.replace(' ?', np.nan)
    df = df.fillna(df.mode().iloc[0])

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')

    x = df.drop(['income'], axis=1)
    y = df['income']

    x = pd.get_dummies(x)
    x = StandardScaler().fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    scaler = StandardScaler()
    scaler.fit(x_train)

    X_train = scaler.transform(x_train)
    X_test = scaler.transform(x_test)

    error = []
    max_val = 25
    for i in range(1, max_val):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, max_val), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='grey', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()

    neighboramount = int(input('Enter the amount of neighbors you want to use: '))

    print('\nNeighborclassifier chosen:', neighboramount, '\n')

    classifier = KNeighborsClassifier(n_neighbors=neighboramount)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def BSP_2():
    df = pd.read_csv('./data/water_potability.csv')
    df = df[(df['ph'] >= 4) & (df['ph'] <= 12)]

    print(df.info())

    df = df.fillna(df.mean())

    x = df.drop('Potability', axis=1)
    y = df['Potability']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    scaler = StandardScaler()
    scaler.fit(x_train)

    X_train = scaler.transform(x_train)
    X_test = scaler.transform(x_test)

    error = []
    max_val = 35
    for i in range(1, max_val):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, max_val), error, color='darkblue', linestyle='dashed', marker='o',
             markerfacecolor='grey', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()

    neighboramount = int(input('Enter the amount of neighbors you want to use: '))

    print('\nNeighborclassifier chosen:', neighboramount, '\n')

    classifier = KNeighborsClassifier(n_neighbors=neighboramount)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Welche Spalte ist für die Labels zuständig?
    # Die Spalte "Potability" ist für die Labels zuständig.

    # Wird One-Hot-Encoding benötigt? Warum?
    # Nein, One-Hot-Encoding wird nicht benötigt, da die Spalte "Potability" bereits gedroppet wird.


def BSP_3():
    df = pd.read_csv('./data/Mall_Customers.csv')

    distortions = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, random_state=55)
        kmeans.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
        distortions.append(kmeans.inertia_)

    plt.plot(range(1, 10), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

    clusteramount = int(input('How many clusters do you want? '))

    kmeans = KMeans(n_clusters=clusteramount, random_state=55)
    kmeans.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    df['cluster'] = kmeans.predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    df['cluster'] = df['cluster'].astype('category')

    plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['cluster'])
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()


def BSP_4():
    df = pd.read_csv('./data/adult.csv')

    df = df.drop(df.columns[2], axis=1)
    df.columns = ['age', 'workclass', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
                  'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

    df = df.drop(df[['education']], axis=1)
    df = df.drop(df[['education_num']], axis=1)
    df = df.drop(df[['relationship']], axis=1)

    df = df.replace(' ?', np.nan)
    df = df.fillna(df.mode().iloc[0])

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')

    x = df.drop(['sex'], axis=1)
    y = df['sex']

    x = pd.get_dummies(x)
    x = StandardScaler().fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    print(str(Counter(y_pred)) + '\n')

    print('Training set score: {:.4f}'.format(gnb.score(x_train, y_train)))
    print('Test set score: {:.4f}'.format(gnb.score(x_test, y_test)) + '\n')

    matrix = confusion_matrix(y_test, y_pred)
    matrix = pd.DataFrame(matrix, index=['Male', 'Female'], columns=['Male', 'Female'])
    print(str(matrix), '\n')

    scores = cross_val_score(gnb, x, y, cv=10)
    print('Cross-validation scores: {}'.format(scores))
    print('Average cross-validation scores: {:.4f}'.format(scores.mean()))

    y_score1 = gnb.fit(x_train, y_train).predict_proba(x_test)[:, 1]
    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1, pos_label=' Male')

    print('roc_auc_score: ', roc_auc_score(y_test, y_score1))
    plt.title('Receiver Operating Characteristic - NB')
    plt.plot(false_positive_rate1, true_positive_rate1)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == '__main__':
    BSP_1()
    BSP_2()
    BSP_3()
    BSP_4()
