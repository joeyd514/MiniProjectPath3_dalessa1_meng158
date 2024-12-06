#MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
#import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import copy


rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

#Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)

def dataset_searcher(number_list,images,labels):
  #insert code that when given a list of integers, will find the labels and images
  #and put them all in numpy arrary (at the same time, as training and testing data)

  # convert to np arrays
  images = np.array(images)
  labels = np.array(labels)

  # use hash set to avoid duplicates
  found_nums = set()
  indices = []

  # find indicies of labels in num list
  for index, label in enumerate(labels):
    if label in number_list and label not in found_nums:
      indices.append(index)
      found_nums.add(label)

    # exit if all nums found
    if len(found_nums) == len(number_list):
      break

  # add found images and labels to arrays
  images_nparray = images[indices]
  labels_nparray = labels[indices]

  return images_nparray, labels_nparray

def print_numbers(images,labels):
  #insert code that when given images and labels (of numpy arrays)
  #the code will plot the images and their labels in the title. 

  # find num images and num rows and cols
  n = len(images)
  rows = int(np.ceil(n / 3))
  cols = min(n, 3)

  # create figure w/ subplots
  figure, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

  # make axes iterable if necessary
  if cols == 1:
    if rows == 1:
      axes = [axes]
    else:
      axes = [[ax] for ax in axes]

  # plot images
  for index, (image, label) in enumerate(zip(images, labels)):
    row = index // 3
    column = index % 3
    
    if rows > 1:
      ax = axes[row][column]
    else:
      ax = axes[column]

    # display images
    ax.imshow(image, cmap='gray')
    ax.axis('off')

  # remove empty subplots
  for index in range(n, rows * cols):
    row = index // 3
    column = index % 3

    if rows > 1:
      ax = axes[row][column]
    else:
      ax = axes[column]

    ax.remove()

  # show plot
  plt.show()

class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher(class_numbers, images, labels)
#Part 2
print_numbers(class_number_images , class_number_labels )


model_1 = GaussianNB()

#however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

#Now we can fit the model
model_1.fit(X_train_reshaped, y_train)
#Part 3 Calculate model1_results using model_1.predict()
model1_results = model_1.predict(X_test.reshape(X_test.shape[0], -1))


def OverallAccuracy(results, actual_values):
  #Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
  
  num_correct = 0

  for i in range(len(results)):
    if results[i] == actual_values[i]:
      num_correct += 1

  Accuracy = num_correct / len(results)

  return Accuracy


# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))


#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)

# reshape and predict using allnumbers
allnumbers_images_reshaped = allnumbers_images.reshape(allnumbers_images.shape[0], -1)
allnumbers_model1_results = model_1.predict(allnumbers_images_reshaped)

# print predicted values using allnumbers
print_numbers(allnumbers_images, allnumbers_model1_results)


# #Part 6
# #Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)


# test comment

# #Repeat for the MLP Classifier
# model_3 = MLPClassifier(random_state=0)



# #Part 8
# #Poisoning
# # Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison


# #Part 9-11
# #Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train




# #Part 12-13
# # Denoise the poisoned training data, X_train_poison. 
# # hint --> Suggest using KernelPCA method from sklearn library, for denoising the data. 
# # When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# # So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64

# X_train_denoised = # fill in the code here


# #Part 14-15
# #Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
# #Explain how the model performances changed after the denoising process.

