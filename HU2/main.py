import pandas as pd
import sys
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve,roc_auc_score
from sklearn.metrics import classification_report


# np.set_printoptions(threshold=sys.maxsize)

def BSP_1():
    df.info()
    df.describe()
    np_1 = np.array(df[['rating']])
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    np_2 = imp_mean.fit_transform(np_1)
    print(np_2)
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    np_2 = imp_median.fit_transform(np_1)
    print(np_2)
    imp_constant = SimpleImputer(missing_values=np.nan, strategy='constant')
    np_2 = imp_constant.fit_transform(np_1)
    print(np_2)
    imp = imputer(missing_values=np.nan, strategy='mean')
    np_3 = imp.fit_transform(np_1)
    print(np_3)
    imp = imputer(missing_values=np.nan, strategy='median')
    np_3 = imp.fit_transform(np_1)
    print(np_3)
    imp = imputer(missing_values=np.nan, strategy='constant')
    np_3 = imp.fit_transform(np_1)
    print(np_3)
    print(df.dropna())


class imputer:
    def __init__(self, missing_values=np.nan, strategy='mean', fill_value=0):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value

    def fit_transform(self, data):
        if math.isnan(self.missing_values):
            if self.strategy == 'mean':
                return np.array(np.nan_to_num(data, nan=np.nanmean(data)))
            elif self.strategy == 'median':
                return np.array(np.nan_to_num(data, nan=np.nanmedian(data)))
            elif self.strategy == 'constant':
                return np.array(np.nan_to_num(data, nan=self.fill_value))
            else:
                raise ValueError('wrong strategy')
        else:
            raise ValueError('only np.nan allowed')


def BSP_2(data, categories):
    nan_list = []
    for (columnName, columnData) in data.items():
        if data[columnName].isnull().values.any() > 0:
            nan_list.append(columnName)
        if columnName in categories:
            data[columnName] = data[columnName].astype('category')
    nan_percent = (data.isnull().sum() / data.size).sum() * 100
    print("Werte mit nan: ",nan_list)
    print("Wie viel Prozent aller Werte sind nan: ",nan_percent)
    print("Spalten mit Category Type: ",data.select_dtypes(include=['category']).columns.values)
    print("Spalten mit Numeric Type: ", data.select_dtypes(exclude=['category']).columns.values)
    cats = {}
    for (dt_name, dt_data) in data.select_dtypes(include=['category']).items():
        tmp = set()
        for d in dt_data:
            for l in str(d).split(', '):
                if l != 'nan':
                    tmp.add(l)
        cats.update({dt_name: tmp})
    print("Werte in den Categorie Spalten: ",cats)


def BSP_3(data):
    df = data.drop(data.columns[2], axis=1)
    df.columns = ['age', 'workclass', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
                  'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    df_1 = df.copy()
    df_2 = df.copy()
    df_3 = df.copy()
    df_1[['capital_gain', 'capital_loss', 'hours_per_week']] = MinMaxScaler().fit_transform(df_1[['capital_gain', 'capital_loss', 'hours_per_week']])
    df_2[['capital_gain', 'capital_loss', 'hours_per_week']] = StandardScaler().fit_transform(df_2[['capital_gain', 'capital_loss', 'hours_per_week']])
    df_3[['capital_gain', 'capital_loss', 'hours_per_week']] = RobustScaler().fit_transform(df_3[['capital_gain', 'capital_loss', 'hours_per_week']])
    print(str(df_1[['capital_gain', 'capital_loss', 'hours_per_week']].describe()))
    print(str(df_2[['capital_gain', 'capital_loss', 'hours_per_week']].describe()))
    print(str(df_3[['capital_gain', 'capital_loss', 'hours_per_week']].describe()))

    # Wie unterscheiden sich die statistischen Eigenschaften der skalierten Daten?
    # Die statistischen Eigenschaften der skalierten Daten unterscheiden sich, da die Skalierungen unterschiedliche
    # Berechnungsmethoden verwenden. Die MinMaxScaler skaliert die Daten auf einen Wertebereich von 0 bis 1, die
    # StandardScaler skaliert die Daten auf einen Mittelwert von 0 und eine Standardabweichung von 1 und die
    # RobustScaler skaliert die Daten auf einen Median von 0 und eine Standardabweichung von 1.

    # Was macht der Robust-Scaler und wann könnte dieser sinnvoll eingesetzt werden?
    # Der Robust-Scaler skaliert die Daten so, dass sie eine Standardabweichung von 1 haben und die Medianwerte 0 sind.
    # Es ist sinnvoll, wenn unsere Daten stark skaliert werden sollen und wir dabei die Ausreißer nicht verlieren wollen.

    # Wann benutzen wir welche Skalierungsmethode?
    # MinMaxScaler:     Wenn wir die Daten auf einen bestimmten Wertebereich skalieren wollen z.B. [0,1]
    # StandardScaler:   Wenn wir die Daten auf eine Standardabweichung von 1 skalieren wollen
    # RobustScaler:     Wenn wir die Daten stark skalieren wollen und dabei die Ausreißer nicht verlieren wollen

    # Welche Skalierungsmethode sollte hier verwendet werden?
    # StandardScaler, da die Daten nicht stark skaliert werden sollen und die Ausreißer nicht verloren gehen sollen.

def BSP_4(data):
    df_1 = data.drop(data.columns[2], axis=1)
    df_1.columns = ['age', 'workclass', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
                 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    df_1.replace({"?": "None"}, inplace=True)
    categories = ['workclass', 'education_num', 'relationship', 'race', 'sex', 'native_country', 'income']
    df_1[categories] = df_1[categories].astype("category")
    feature = df_1.drop(["income"], axis=1)
    target = df_1['income']
    feature = pd.get_dummies(feature)
    cols = feature.columns
    feature = StandardScaler().fit_transform(feature)
    feature = pd.DataFrame(feature, columns=cols)
    x_train, x_test, y_train, y_test = train_test_split(feature, target,test_size=0.8,random_state=1401)
    print(len(x_train), len(y_test), len(x_train), len(y_test))
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    print(y_pred.shape)
    print('Training Ergebnis: {:.4f}'.format(gnb.score(x_train, y_train)))
    print('Test Ergebnis: {:.4f}'.format(gnb.score(x_test, y_test)))
    # BSP5
    print("Falsch gelabelte Kunden aus insgesamt %d Kunden : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
    print("Falsch gelabelte Kunden:", ((y_test != y_pred).sum()) / x_test.shape[0], "%")
    accuracy = accuracy_score(y_test, y_pred) * 100
    print("Genauigkeit: ", accuracy)
    matrix = confusion_matrix(y_test, y_pred)
    matrix = pd.DataFrame(matrix, index=['Positive', 'Negative'], columns=['Positive', 'Negative'])
    print(classification_report(y_test, y_pred))
    y_score1 = gnb.fit(x_train, y_train).predict_proba(x_test)[:, 1]
    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1, pos_label=' >50K')
    print('roc_auc_score: ', roc_auc_score(y_test, y_score1))
    plt.title('Receiver Operating Characteristic - NB')
    plt.plot(false_positive_rate1, true_positive_rate1)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    y_true = [0, 1, 2, 2, 2]
    y_pred = [0, 0, 2, 2, 1]
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(y_true, y_pred, target_names=target_names))
    y_pred = [1, 1, 0]
    y_true = [1, 1, 1]
    print(classification_report(y_true, y_pred, labels=[1, 2, 3]))


if __name__ == '__main__':
    df = pd.read_csv('./data/anime.csv')
    df_2 = pd.read_csv('./data/scrubbed.csv')
    df_3 = pd.read_csv('./data/adult.csv')
    BSP_1()
    BSP_2(df_2, categories=['shape'])
    BSP_3(df_3)
    BSP_4(df_3)
