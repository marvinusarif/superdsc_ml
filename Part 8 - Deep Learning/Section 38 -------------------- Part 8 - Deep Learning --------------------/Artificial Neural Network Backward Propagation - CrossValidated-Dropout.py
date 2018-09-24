import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

class testnjob() :
    def __init__(self, X_train, y_train) :
        classifier = KerasClassifier(build_fn = self.build_classifier,
                             batch_size=10,
                             nb_epoch = 3)
        self.accuracies = cross_val_score(estimator = classifier,
                             X = X_train,
                             y = y_train,
                             cv = 10,
                             n_jobs=-1)

    def build_classifier(self):
        classifier = Sequential()
        classifier.add(Dense(input_dim=11,
                             units=6,
                             kernel_initializer='uniform',
                             activation = 'relu'))
        classifier.add(Dense(units=6,
                             kernel_initializer='uniform',
                             activation="relu"))
        classifier.add(Dense(units=1,
                             kernel_initializer='uniform',
                             activation="sigmoid"))
        classifier.compile(optimizer="adam",
                       loss = "binary_crossentropy",
                       metrics = ["accuracy"])
        return classifier

    def message_print(self,msg):
        print ("##################################")
        print (msg)
        print ("##################################")

    def print_accuracies(self):

        self.message_print('ACCURACIES:'+str(self.accuracies))
        self.message_print('MEAN:' + str(np.mean(self.accuracies)))
        self.message_print('STD:' + str(np.std(self.accuracies)))

if __name__ == "__main__":
    df = pd.read_csv("../Section 39 - Artificial Neural Networks (ANN)/Churn_Modelling.csv")
    df.tail()

    _df = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']]
    _df.shape

    X = _df.iloc[:, :-1].values
    y = _df.iloc[:, -1].values

    X.shape, X[:3], y.shape, y[:3]

    _df['Geography'].value_counts()

    label_encoder = LabelEncoder()
    X[:, 1] = label_encoder.fit_transform(X[:,1])
    X[:]

    X[:, 2] = label_encoder.fit_transform(X[:, 2])
    X[:]

    hot_encoder = OneHotEncoder(categorical_features=[1])
    X = hot_encoder.fit_transform(X).toarray()
    X.shape, X[0]

    #remove dummy variables
    X = X[:, 1:]
    X.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train.shape, y_train.shape, X_test.shape, y_test.shape

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # DO preprocessing here and get X_train and y_train
    test_obj = testnjob(X_train, y_train)
    test_obj.print_accuracies()
