import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import seaborn as sns


# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index)) # shuffle the examples
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
# Calculate the Z-scores of each column in the training set:
train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean)/train_df_std
# Calculate the Z-scores of each column in the test set.
test_df_mean = test_df.mean()
test_df_std = test_df.std()
test_df_norm = (test_df - test_df_mean)/test_df_std
# Create an empty list that will eventually hold all created feature columns.
feature_columns = []
# We scaled all the columns, including latitude and longitude, into their
# Z scores. So, instead of picking a resolution in degrees, we're going
# to use resolution_in_Zs.  A resolution_in_Zs of 1 corresponds to
# a full standard deviation.
resolution_in_Zs = 0.3  # 3/10 of a standard deviation.
# Create a bucket feature column for latitude.
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arange(int(min(train_df_norm['latitude'])),
                                    int(max(train_df_norm['latitude'])),
                                    resolution_in_Zs))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, latitude_boundaries)
# Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df_norm['longitude'])),
                                     int(max(train_df_norm['longitude'])),
                                     resolution_in_Zs))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column,
                                               longitude_boundaries)
# Create a feature cross of latitude and longitude.
latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature) 
# Represent median_income as a floating-point value.
median_income = tf.feature_column.numeric_column("median_income")
feature_columns.append(median_income)
# Represent population as a floating-point value.
population = tf.feature_column.numeric_column("population")
feature_columns.append(population)
# Convert the list of feature columns into a layer that will later be fed into
# the model.
my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


def plot_the_loss_curve(epochs, mse):
 """Plot a curve of loss vs. epoch."""


 plt.figure()
 plt.xlabel("Epoch")
 plt.ylabel("Mean Squared Error")
 plt.plot(epochs, mse, label="Loss")
 plt.legend()
 plt.ylim([mse.min()*0.95, mse.max() * 1.03])
 plt.show() 
       
def create_model(my_learning_rate, my_feature_layer):
 """Create and compile a simple linear regression model."""


 # Most simple tf.keras models are sequential.
 model = tf.keras.models.Sequential()
 # Add the layer containing the feature columns to the model.
 model.add(my_feature_layer)
 # Define the first hidden layer with 20 nodes.  
 model.add(tf.keras.layers.Dense(units=10,
                                 activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                 name='Hidden1'))
 model.add(tf.keras.layers.Dropout(rate=0.25));
 # Define the second hidden layer with 12 nodes.
 model.add(tf.keras.layers.Dense(units=6,
                                 activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                 name='Hidden2'))
  #model.add(tf.keras.layers.Dropout(rate=0.1));
 # Define the output layer.
 model.add(tf.keras.layers.Dense(units=1, 
                                 name='Output'))                             
 model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
               loss="mean_squared_error",
               metrics=[tf.keras.metrics.MeanSquaredError()])
 return model


def train_model(model, dataset, epochs, batch_size, label_name):
 """Feed a dataset into the model in order to train it."""


 # Split the dataset into features and label.
 features = {name:np.array(value) for name, value in dataset.items()}
 label = np.array(features.pop(label_name))
 history = model.fit(x=features, y=label, batch_size=batch_size,
                     epochs=epochs, shuffle=True)
 # Get details that will be useful for plotting the loss curve.
 epochs = history.epoch
 hist = pd.DataFrame(history.history)
 rmse = hist["mean_squared_error"]
 return epochs, rmse  

# The following variables are the hyperparameters.
learning_rate = 0.05
epochs = 30
batch_size = 1000
label_name = "median_house_value"
# Establish the model's topography.
my_model = create_model(learning_rate, my_feature_layer)
# Train the model on the normalized training set.
epochs, mse = train_model(my_model, train_df_norm, epochs, batch_size, label_name)
plot_the_loss_curve(epochs, mse)
test_features = {name:np.array(value) for name, value in test_df_norm.items()}
test_label = np.array(test_features.pop(label_name)) # isolate the label
my_model.evaluate(x = test_features, y = test_label, batch_size=batch_size)
