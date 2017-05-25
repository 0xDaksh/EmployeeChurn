import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
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
from keras.models import model_from_json

# creating the model
clf = Sequential([
    Dense(units=11, kernel_initializer='uniform', activation='relu', input_dim=10),
    Dense(units=11, kernel_initializer='uniform', activation='relu'), # units are based on my creativity :3
    Dense(1, kernel_initializer='uniform', activation='sigmoid') #output
])

# compiling the model
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# training the model
clf.fit(X_train, Y_train, batch_size=9, epochs=100)

# Evaluate model
score = clf.evaluate(X_test, Y_test, batch_size=128)
print(score[1]*100) # 96.9333337148% or 0.96133

# serialize model to JSON
clf_json = clf.to_json()
with open("clf.json", "w") as json_file:
    json_file.write(clf_json)
# serialize weights to HDF5
clf.save_weights("clf.h5")
print("Saved model to disk")

# load json and create model
json_file = open('clf.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loadedClf = model_from_json(loaded_model_json)
# load weights into new model
loadedClf.load_weights("clf.h5")
print("Loaded model from disk")

loadedClf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
score = loadedClf.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (loadedClf.metrics_names[1], score[1]*100)) # 96.93
