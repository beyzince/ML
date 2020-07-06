# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# dosyalardan veriler alınır
data1 = pd.read_csv("/Users/onno/Desktop/beyzaince/hotel_bookings.csv",usecols =['hotel','is_canceled','lead_time','arrival_date_year','arrival_date_month','arrival_date_week_number','arrival_date_day_of_month','is_repeated_guest','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies','meal','reserved_room_type','assigned_room_type','reservation_status','reservation_status_date','country'])

#print(data1.head())

#satir sutun sayisi alindi
print(data1.shape)

#tablo bilgileri
#print(data1.info)

# nan değerler kontrol edildi.
print(data1.isnull().sum())

# nan değerler silindi.
data1 = data1.dropna(axis=0)


#Son Satır Sutun Durum

print(data1.shape)
#print(data1.dtypes)
'''

colors = ['r', 'g', 'b', 'c', 'm']
numbers = data1["hotel"].value_counts()
paths = data1["hotel"].value_counts().keys()
plt.title("otel turu")
plt.ylabel('count')
plt.bar(paths, numbers,color=colors)
plt.savefig('otel.png')
#is_canceled
numbers = data1["is_canceled"].value_counts()
paths = data1["is_canceled"].value_counts().keys()
plt.title("iptal durumu")

plt.ylabel('count')
plt.bar(paths, numbers,color=colors)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('iptaldurumu.png')
'''


'''
numbers = data1["lead_time"].value_counts()
paths = data1["lead_time"].value_counts().keys()
plt.title("gecen gun sayisi")
plt.ylabel('count')
plt.bar(paths, numbers,color=colors)
plt.savefig('gunsayisi.png')

numbers = data1["adults"].value_counts()
paths = data1["adults"].value_counts().keys()
plt.title("yetiskin")
plt.ylabel('count')
plt.bar(paths, numbers,color=colors)
plt.savefig('yetiskin.png')


numbers = data1["arrival_date_week_number"].value_counts()
paths = data1["arrival_date_week_number"].value_counts().keys()
plt.title("hafta varis tarihi icin yil sayisi")
plt.ylabel("count")
plt.bar(paths, numbers,color=colors)
plt.legend()
plt.tight_layout()
plt.savefig("yil.png")
'''




#veri sayisallastirma
le = preprocessing.LabelEncoder()
dtype_object=data1.select_dtypes(include=['object'])
for x in dtype_object.columns:
    data1[x]=le.fit_transform(data1[x])
print(data1.dtypes)

X = data1.iloc[:,1:19]
y = data1['hotel'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X.head())
#print(y.head())


#sayisal verileri olceklendirdik
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



def matrisal(y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred, labels=np.unique(y_pred))
    print('Doğruluk Oranı: %' + str(ac * 100))
    print('**********************************\nKonfüzyon Matrisi:')
    print(cm)
    print('************************************\nF1, Recall, Precision Skorları Tablosu:')
    print(cr)
    

# lojistik regresyon ile eğitim


lr = LogisticRegression(C=1e5)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
matrisal(y_pred, y_test)

#svm ile eğitim

svm = SVC(kernel="linear")
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
matrisal(y_pred, y_test)

#  knn ile eğitim

knn = KNeighborsClassifier(n_neighbors=244, algorithm='kd_tree')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
matrisal(y_pred, y_test)

#  naive bayes ile eğitim

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
matrisal(y_pred, y_test)

# karar ağacı ile eğitim

tree = DecisionTreeClassifier(criterion='entropy', max_depth=120)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
matrisal(y_pred, y_test)

#yapay sinir ağıyla eğitim

ann = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10, 4), random_state=1)
ann.fit(X_train, y_train)
y_pred = ann.predict(X_test)
matrisal(y_pred, y_test)











