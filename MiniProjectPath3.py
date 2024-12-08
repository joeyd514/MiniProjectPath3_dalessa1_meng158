#MiniProjectPath3
import warnings
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.decomposition import KernelPCA, PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
#import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import copy

warnings.filterwarnings('ignore', category=ConvergenceWarning)

rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

#Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)

# print(f"X_train shape: {X_train.shape}")
# print(f"X_test shape: {X_test.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"y_test shape: {y_test.shape}")

def dataset_searcher(number_list, images, labels):
    #insert code that when given a list of integers, will find the labels and images
    #and put them all in numpy arrary (at the same time, as training and testing data)

    mask = np.isin(labels, number_list)
    images_nparray = images[mask]
    labels_nparray = labels[mask]
    return images_nparray, labels_nparray


def print_numbers(images, labels, max_images=100, title="original"):
    #insert code that when given images and labels (of numpy arrays)
    #the code will plot the images and their labels in the title.
    num_images = min(len(images), max_images)
    cols = 10
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=(cols * 0.8, rows * 0.8))
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 8))
    for img, label, ax in zip(images, labels, axs.ravel()):
        ax.imshow(img.reshape((8,8)), cmap="Greys")
        ax.set_title(f"{label}", fontsize=15, pad=5)
        ax.axis("off")
    fig.subplots_adjust(hspace=1)
    fig.suptitle(title, fontsize=24)
    plt.tight_layout()
    plt.show()

class_numbers = [2, 0, 8, 7, 5]
#Part 1
class_number_images, class_number_labels = dataset_searcher(class_numbers, images, labels)
#Part 2
print_numbers(class_number_images, class_number_labels, title="Original - [2,0,8,7,5]")


model_1 = GaussianNB()

#however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
# print(f"X_train_reshaped shape: {X_train_reshaped.shape}")

#Now we can fit the model
model_1.fit(X_train_reshaped, y_train)

#Part 3 Calculate model1_results using model_1.predict()
X_test_reshape = X_test.reshape(X_test.shape[0], -1)
model1_results = model_1.predict(X_test_reshape)


def OverallAccuracy(results, actual_values):
    #Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
    Accuracy = accuracy_score(actual_values, results) * 100
    return Accuracy


# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print(f"The overall results(Original) of the Gaussian model is {Model1_Overall_Accuracy:.2f}%")


def run_model(model_detail, train_data, test_data, allnumbers_data, description="Original"):
    model_name, model = model_detail
    X_train_image, y_train_label = train_data
    X_test_image, y_test_label = test_data
    all_numbers_image, all_numbers_label = allnumbers_data
    model.fit(X_train_image, y_train_label)
    results = model.predict(X_test_image)
    accuracy = OverallAccuracy(results, y_test_label)
    print(f"The overall results({description}) of the {model_name} is {accuracy:.2f}%")
    allnumbers_results = model.predict(all_numbers_image)
    print_numbers(allnumbers_images, allnumbers_results, title=f"{description} - {model_name}")

models = OrderedDict({
    "Gaussian": ["Gaussian model", GaussianNB()],
    "KNeighbors": ["K Nearest Neighbors", KNeighborsClassifier(n_neighbors=10)],
    "MLP": ["MLP", MLPClassifier(random_state=0)],
})

#Part 5
allnumbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)
allnumbers_reshaped = allnumbers_images.reshape(allnumbers_images.shape[0], -1)

model1_results = model_1.predict(allnumbers_reshaped)
print_numbers(allnumbers_images, model1_results, title="Original - Gaussian model")

#Part 6
train_data = [X_train_reshaped, y_train]
test_data = [X_test_reshape, y_test]
print_data = [allnumbers_reshaped, allnumbers_labels]

#Repeat for K Nearest Neighbors
run_model(models["KNeighbors"], train_data, test_data, print_data)

#Repeat for the MLP Classifier
run_model(models["MLP"], train_data, test_data, print_data)

#Part 8
#Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison

#Part 9-11
#Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train
X_train_poison_reshaped = X_train_poison.reshape(X_train_poison.shape[0], -1)

train_data = [X_train_poison_reshaped, y_train]
test_data = [X_test_reshape, y_test]
print_data = [allnumbers_reshaped, allnumbers_labels]

models_list = [model_detail for name, model_detail in models.items()]
for model_detail in models_list:
    run_model(model_detail, train_data, test_data, print_data, "Poisoned")

#Part 12-13
# Denoise the poisoned training data, X_train_poison.
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data.
# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64

# X_train_denoised = # fill in the code here
kpca = KernelPCA(
    n_components=20,
    kernel="rbf",
    gamma=1e-4,
    fit_inverse_transform=True,
    alpha=1e-3,
    random_state=42,
)

kpca.fit(X_train_poison_reshaped)
X_train_denoised = kpca.inverse_transform(kpca.transform(X_train_poison_reshaped))
X_test_denoised = kpca.inverse_transform(kpca.transform(X_test_reshape))
allnumbers_denoised = kpca.inverse_transform(kpca.transform(allnumbers_reshaped))

# print_numbers(X_train, y_test, title="original data")
# print_numbers(X_train_poison, y_test, title="poisoned data")
# print_numbers(X_train_denoised, y_test, title="denoised data")

#Part 14-15
#Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
#Explain how the model performances changed after the denoising process.
train_data = [X_train_denoised, y_train]
test_data = [X_test_denoised, y_test]
print_data = [allnumbers_denoised, allnumbers_labels]

for model_detail in models_list:
    run_model(model_detail, train_dat, test_data, print_data, "Denoised")
