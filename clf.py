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
