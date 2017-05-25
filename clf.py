import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

#fix for memory allocation ERROR
def get_session(gpu_fraction=0.5):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())
# Data Pre preprocessing
df = pd.read_csv('dataset.csv')
X = df.iloc[:, 0:9].values
y = df.iloc[:, 9:10].values

# label encoding
le = LabelEncoder()
X[:, -1] = le.fit_transform(X[:, -1])
X[:, -2] = le.fit_transform(X[:, -2])
# One hot encoding
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
# Creating sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=1)
# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Libraries
from keras.models import Sequential
from keras.layers import Dense
# creating the model
clf = Sequential([
    Dense(units=8, kernel_initializer='uniform', activation='relu', input_dim=10),
    Dense(units=9, kernel_initializer='uniform', activation='relu'), # units are based on my creativity :3
    Dense(1, kernel_initializer='uniform', activation='sigmoid') #output
])

# compiling the model
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# training the model
clf.fit(X_train, Y_train, batch_size=9, epochs=100)

# predicting the values
Y_pred = clf.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Evaluate model
score = clf.evaluate(X_test, Y_test, batch_size=128)
print(score) # 95% or 0.9586
