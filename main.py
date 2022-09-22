# Import libraries
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# File paths of Xray images
mydir = "chest_xray/"
train = 'chest_xray/train/'
test = 'chest_xray/test/'
val = 'chest_xray/val/'

train_dir = os.path.join(mydir, 'train')
test_dir = os.path.join(mydir, 'test')
val_dir = os.path.join(mydir, 'val')

# Train directory
train_normal_dir = os.path.join(train_dir, 'NORMAL')
train_pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')

print("Normal images in train: ")
print(len(os.listdir(train_normal_dir)))
print("Pneumonia images in train: ")
print(len(os.listdir(train_pneumonia_dir)))

# Test directory
test_normal_dir = os.path.join(test_dir, 'NORMAL')
test_pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')

print("Normal images in test: ")
print(len(os.listdir(test_normal_dir)))
print("Pneumonia images in test: ")
print(len(os.listdir(test_pneumonia_dir)))

# Validation directory
val_normal_dir = os.path.join(val_dir, 'NORMAL')
val_pneumonia_dir = os.path.join(val_dir, 'PNEUMONIA')

print("Normal images in val: ")
print(len(os.listdir(val_normal_dir)))
print("Pneumonia images in val: ")
print(len(os.listdir(val_pneumonia_dir)))

# Standardise all images height and width
img_height = 500
img_width = 500

# Batch size for testing
batch_size = 16

# Using Image Data Generator artificially increase the size of the data set

image_generate = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_data = image_generate.flow_from_directory(
    directory=train_dir,
    batch_size=batch_size,
    target_size=(img_height, img_width),
    class_mode='binary',
    color_mode='grayscale'
)

test_data = image_generate.flow_from_directory(
    directory=test_dir,
    batch_size=batch_size,
    target_size=(img_height, img_width),
    shuffle=False,
    class_mode='binary',
    color_mode='grayscale'
)

val_data = image_generate.flow_from_directory(
    directory=val_dir,
    batch_size=batch_size,
    target_size=(img_height, img_width),
    class_mode='binary',
    color_mode='grayscale'
)

# Use numpy and sklearn to compute the class weights for each class so that the model can learn from each class equally
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
wt = dict(zip(np.unique(train_data.classes), weights))
print('weights', wt)

# import keras to create and compile the model
model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_width, img_height, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_width, img_height, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(img_width, img_height, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(img_width, img_height, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(activation='relu', units=128),
    Dense(activation='relu', units=64, ),
    Dense(activation='sigmoid', units=1)
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# Display the model
model.summary()

# import early stopping to stop the model from over fitting
early = EarlyStopping(monitor='val_loss', mode='min', patience=3)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=2, verbose=1, factor=0.3, min_lr=0.000001)
callbacks_list = [early, learning_rate_reduction]

# model fit containing all the parameters including number of epochs class weight etc
model.fit(
    train_data,
    epochs=15,
    validation_data=test_data,
    class_weight=wt,
    callbacks=callbacks_list
)

# import plotly graph to display graph of accuracies against epochs
graph = go.Figure()
graph.add_trace(go.Scatter(
    x=model.history.epoch,
    y=model.history.history['accuracy'],
    mode='lines+markers',
    name='Training accuracy')
)
graph.add_trace(go.Scatter(
    x=model.history.epoch,
    y=model.history.history['val_accuracy'],
    mode='lines+markers',
    name='Validation accuracy')
)
graph.update_layout(
    title='Accuracy',
    xaxis=dict(title='Epoch'),
    yaxis=dict(title='Percentage'))
graph.show()

# display testing accuracy
test_accu = model.evaluate(test_data)
print('The testing accuracy is :', test_accu[1] * 100, '%')

# designate predictions either 1 or 0 by evaluating everything less than .5 as 0 and everything over as 1
preds = model.predict(test_data, verbose=1)
predictions = preds.copy()
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1

# Display confusion matrix
cm = pd.DataFrame(data=confusion_matrix(test_data.classes, predictions, labels=[0, 1]),
                  index=["Actual Normal", "Actual Pneumonia"],
                  columns=["Predicted Normal", "Predicted Pneumonia"])
sns.heatmap(cm, annot=True, fmt="d")

# print classification report
print(classification_report(y_true=test_data.classes, y_pred=predictions, target_names=['NORMAL', 'PNEUMONIA']))

# Extract images from test data from iterator without shuffling.
test_data.reset()
x = np.concatenate([test_data.next()[0] for i in range(test_data.__len__())])
y = np.concatenate([test_data.next()[1] for i in range(test_data.__len__())])
print(x.shape)
print(y.shape)

# import matplot to display 9 random xrays with results from model
# x = images  Y = labels
dic = {0: 'NORMAL', 1: 'PNEUMONIA'}
plt.figure()
for i in range(0 + 230, 9 + 230):
    plt.subplot(3, 3, (i - 230) + 1)
    if preds[i, 0] >= 0.5:
        out = ('{:.2%} probability of being PNEUMONIA case'.format(preds[i][0]))
        plt.title(out + "\n Actual case : " + dic.get(y[i]))

    else:
        out = ('{:.2%} probability of being NORMAL case'.format(1 - preds[i][0]))
        plt.title(out + "\n Actual case : " + dic.get(y[i]))
    plt.imshow(np.squeeze(x[i]), cmap='gray')
    plt.axis('off')

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()

# Testing with a ValidateChest X-Ray
check_path = 'chest_xray/val/pneumonia/val-pneumonia (14).jpeg'

check_img = image.load_img(check_path, target_size=(img_height, img_width, 1),
                           color_mode='grayscale')
# Process the image
pro_check_img = image.img_to_array(check_img)
pro_check_img = pro_check_img / 255
pro_check_img = np.expand_dims(pro_check_img, axis=0)
check_preds = model.predict(pro_check_img)
plt.figure()
ptitle = 'Test Xray ' + check_path
plt.suptitle(ptitle, fontsize=16)
plt.axis('off')
if check_preds >= 0.5:
    out = (' {:.2%} percent that this is a PNEUMONIA case'.format(check_preds[0][0]))

else:
    out = (' {:.2%} percent that this is a NORMAL case'.format(1 - check_preds[0][0]))
plt.title("Check chest X-Ray\n" + out)
plt.imshow(np.squeeze(pro_check_img), cmap='gray')
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()
