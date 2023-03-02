import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt


# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
# tf.keras.backend.set_floatx('float32')

train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index)) # shuffle the training set
# Calculate the Z-scores of each column in the training set and
# write those Z-scores into a new pandas DataFrame named train_df_norm.
train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean)/train_df_std
# Calculate the Z-scores of each column in the test set and
# write those Z-scores into a new pandas DataFrame named test_df_norm.
test_df_mean = test_df.mean()
test_df_std  = test_df.std()
test_df_norm = (test_df - test_df_mean)/test_df_std
# We arbitrarily set the threshold to 265,000, which is
# the 75th percentile for median house values.  Every neighborhood
# with a median house price above 265,000 will be labeled 1,
# and all other neighborhoods will be labeled 0.
threshold = 265000
train_df_norm["median_house_value_is_high"] = (train_df["median_house_value"] > threshold).astype(float)
test_df_norm["median_house_value_is_high"] = (test_df["median_house_value"] > threshold).astype(float)
train_df_norm["median_house_value_is_high"].head(8000)
# Alternatively, instead of picking the threshold
# based on raw house values, you can work with Z-scores.
# For example, the following possible solution uses a Z-score
# of +1.0 as the threshold, meaning that no more
# than 16% of the values in median_house_value_is_high
# will be labeled 1.
# threshold_in_Z = 1.0
# train_df_norm["median_house_value_is_high"] = (train_df_norm["median_house_value"] > threshold_in_Z).astype(float)
# test_df_norm["median_house_value_is_high"] = (test_df_norm["median_house_value"] > threshold_in_Z).astype(float)
feature_columns = []
# Create a numerical feature column to represent median_income.
median_income = tf.feature_column.numeric_column("median_income")
feature_columns.append(median_income)
# Create a numerical feature column to represent total_rooms.
tr = tf.feature_column.numeric_column("total_rooms")
feature_columns.append(tr)
# Convert the list of feature columns into a layer that will later be fed into
# the model.
feature_layer = layers.DenseFeatures(feature_columns)
# Print the first 3 and last 3 rows of the feature_layer's output when applied
# to train_df_norm:
feature_layer(dict(train_df_norm))

def create_model(my_learning_rate, feature_layer, my_metrics):
 """Create and compile a simple classification model."""


 # Most simple tf.keras models are sequential.
 model = tf.keras.models.Sequential()
 # Add the feature layer (the list of features and how they are represented)
 # to the model.
 model.add(feature_layer)
 # Funnel the regression value through a sigmoid function.
 model.add(tf.keras.layers.Dense(units=1, input_shape=(1,),
                                 activation=tf.sigmoid),)
 # Call the compile method to construct the layers into a model that
 # TensorFlow can execute.  Notice that we're using a different loss
 # function for classification than for regression.   
 model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),                                                  
               loss=tf.keras.losses.BinaryCrossentropy(),
               metrics=my_metrics)
 return model       

def train_model(model, dataset, epochs, label_name,
               batch_size=None, shuffle=True):
 """Feed a dataset into the model in order to train it."""


 # The x parameter of tf.keras.Model.fit can be a list of arrays, where
 # each array contains the data for one feature.  Here, we're passing
 # every column in the dataset. Note that the feature_layer will filter
 # away most of those columns, leaving only the desired columns and their
 # representations as features.
 features = {name:np.array(value) for name, value in dataset.items()}
 label = np.array(features.pop(label_name))
 history = model.fit(x=features, y=label, batch_size=batch_size,
                     epochs=epochs, shuffle=shuffle)
  # The list of epochs is stored separately from the rest of history.
 epochs = history.epoch
 # Isolate the classification metric for each epoch.
 hist = pd.DataFrame(history.history)
 return epochs, hist 

def plot_curve(epochs, hist, list_of_metrics):
 """Plot a curve of one or more classification metrics vs. epoch.""" 
 # list_of_metrics should be one of the names shown in:
 # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics 


 plt.figure()
 plt.xlabel("Epoch")
 plt.ylabel("Value")
 for m in list_of_metrics:
   x = hist[m]
   plt.plot(epochs[1:], x[1:], label=m)
 plt.legend()

# The following variables are the hyperparameters.
learning_rate = 0.001
epochs = 20
batch_size = 100
classification_threshold = 0.52
label_name = "median_house_value_is_high"
# Here is the updated definition of METRICS:
METRICS = [
     tf.keras.metrics.BinaryAccuracy(name='accuracy',
                                     threshold=classification_threshold),
     tf.keras.metrics.Precision(thresholds=classification_threshold,
                                name='precision'
                                ),
     tf.keras.metrics.Recall(thresholds=classification_threshold,
                             name="recall"),
     tf.keras.metrics.AUC(num_thresholds=100,
                             name="auc"),
]
# Establish the model's topography.
my_model = create_model(learning_rate, feature_layer, METRICS)
# Train the model on the training set.
epochs, hist = train_model(my_model, train_df_norm, epochs,
                          label_name, batch_size)
# Plot metrics vs. epochs
list_of_metrics_to_plot = ['accuracy', "precision", "recall", "auc"]
plot_curve(epochs, hist, list_of_metrics_to_plot)
# A `classification_threshold` of slightly over 0.5
# appears to produce the highest accuracy (about 83%).
# Raising the `classification_threshold` to 0.9 drops
# accuracy by about 5%.  Lowering the
# `classification_threshold` to 0.3 drops accuracy by
# about 3%.
features = {name:np.array(value) for name, value in test_df_norm.items()}
label = np.array(features.pop(label_name))
my_model.evaluate(x = features, y = label, batch_size=batch_size)
