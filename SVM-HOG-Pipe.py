# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 09:39:40 2019

@author: yanivmal
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:35:52 2019

@author: yanivmal
"""

import numpy as np
import cv2
import os
import skimage as ski
import matplotlib as plt
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def GetDefaultParameters():
#function that returns a list of the parameters
    params = {}
    params['data_path'] = "C:/Users/dsalt/OneDrive/Desktop/ObjectCategories"
    # indices start from 0 -> i.e for classes 1,2,3 the indices are 0,1,2!
    params['class_indices'] = [10,11,12,13,14,15,16,17,18,19]
    params['S'] = 93
    params['ratio'] = 20
    params['num_bins'] = 13  # bins for HOG
    params['pixels_per_cell'] = 13
    params['cells_per_block'] = 2
    params['max_iter'] = 1000  # iteration fot the svm
    params['C'] = 10
    params['block_norm'] = 'L2-Hys'
    params['gamma'] = 0.96  # for the RBF kernel
    return params

def GetData(data_path, S, class_indices, ratio):
# loading the data from the path
# param: data_path: string- address of the folder with the photos
# param: S: int- the size of the images
# param: class_indices: vector of indices of the classes that we want to take photos from
# param: ratio: int that describes the number of images for the train set
# output: finaldata: matrix of the data after the wanted manipulation
# output: labels: vector of the labels of the images
# output: size: vector which indicates the size of the classes in serial numbers
   finaldata = []
   labels = []
   size = []
   data = os.listdir(data_path)
   classes = 0
   for i in class_indices:
       in_path = data_path + '/' + data[i]
       images = os.listdir(in_path)
       images.sort()
       index = min(ratio*2, len(images))
       size.append(index)
       for j in range(index):
           image = images[j]
           pathi = in_path + '/' + image
           temp = cv2.imread(pathi)
           grayImage = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
           finalImage = cv2.resize(grayImage, (S, S))
           finaldata.append(finalImage)
           labels.append(classes)
       classes = classes + 1
   return finaldata, labels, size


def TrainTestSplit(data, labels, size, class_indices, ratio):
# function that split the data to ttrain and test sets
# param: data: matrix of the images
# param: labels: vector of labels of the images
# param: size: vector which indicates the size of the classes in serial numbers
# param: class_indices: vector of indices of the classes that we want to take photos from
# param: ratio: int that describe the number of images for the train set
# output: trainData: matrix of images for the train set
# output: testData: matrix of images for the test set
# output: trainLabels: vector of labels of the train set
# output: testLabels: vector of labels of the test set
   index = 0
   trainData = []
   testData = []
   TrainLabels = []
   TestLabels = []
   for i in range(len(class_indices)):
       trainData.extend(data[index:index + ratio])
       testData.extend(data[index + ratio:index + size[i]])
       TrainLabels.extend(labels[index:index + ratio])
       TestLabels.extend(labels[index + ratio:index + size[i]])
       index = index + size[i]
   return trainData, testData, TrainLabels, TestLabels


def Prepare(data, num_bins, pixels_per_cell, cells_per_block,b):
#function that prepare each image to a vector for the hog function
# param: data: matrix of the images
# param: num_bins: int that describe the number of oriented gradients
# param: pixels_per_cell: int
# param: cells_per_block: int
# output: dataHog: list of vectors that ready for the svm algorithm
   dataHog = []
   for image in data:
       image_feature = ski.feature.hog(image, orientations=num_bins, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                       block_norm=b ,cells_per_block=(cells_per_block, cells_per_block))
       dataHog.append(image_feature)
   return dataHog


#def Train(data, labels, c, max_iter, flag):
# function that train the classifier
# param: data: matrix of the the observations (vectors of the images)
# param: labels: vector of the labels of the data
# param: c: number that is a parameter for the SVM algorithm. tradeoff between complexity and fit.
# param: max_iter: int that describe the maximum number of iterations of the SVM algorithm
#output: linear_clf: list that described the trained classifier.
#    if flag:
#        linear_clf = LinearSVC(C=c, multi_class='ovr', verbose=0, random_state=None, max_iter=max_iter)
#    else:
#        clf = SVC(C=c, multi_class='ovr', verbose=0, random_state=None, max_iter=max_iter)
#    linear_clf.fit(data, labels)
#    return linear_clf


def n_class_SVM_train(data, class_indices, c, max_iter, ratio, labels):
# function that train the classifier according to M classes
# param: data: matrix of the the observations (vectors of the images)
# param: labels: vector of the labels of the data
# param: c: number that is a parameter for the SVM algorithm. tradeoff between complexity and fit.
# param: max_iter: int that describe the maximum number of iterations of the SVM algorithm
# output: clf: list that described the trained classifier.
   clf = SVC(C=c, kernel='linear', decision_function_shape='ovr', verbose=0, random_state=10, max_iter=max_iter)
   clf.fit(data, labels)
   #    for i in range(len(class_indices)):
   #      Tlabels=np.ones(len(data))*-1
   #      Tlabels[i*ratio:(i+1)*ratio]=1
   #      linear_clf.append(Train(data,Tlabels,c,max_iter))
   return clf

def n_class_SVM_predict(data, class_indices, clf):
# function that test the classifier on given data
# param: data: matrix of the the observations (vectors of the images)
# param: model: list that described the trained classifier
# output: Results: vector of the predicted labels
   predict = clf.decision_function(data)
   Results = np.argmax(predict, axis=1)
   #    for i in range(len(class_indices)):
   #        scoreMatrix.append(test(data,linear_models[i]))
   #    scoreMatrix= np.reshape(scoreMatrix, (len(class_indices),len(data)))
   #    Results= np.argmax(scoreMatrix,axis=0)
   return Results, predict


def Evaluate(Results, labels):
#function that evaluate the results on the test data
#param: Results: vector of the predicted labels
#param: labels: vector of the real labels
#output: confusion_matrix: matrix of size M×M.
#output: err_rate: the rate of the error
   err_rate = sum(Results != labels) / len(labels)
   return confusion_matrix(Results, labels), err_rate


def kFoldsPartition(TrainLabels, TrainData, folds, ratio, class_indices):
#function that divide the train data to folds for the k-Fold (cross validation)
#param: TrainLabels: vector of the labels of the train set
#param: TrainData: matrix of the train set
#param: folds: number of folds
#param: ratio: int that describes the number of images for the train set
#param: class_indices: the classes
#output: Kfolds: matrix of the train set that divided to k-folds.
#output: labelsfolds: the labels of the train set divided to k-folds
   Kfolds = []
   data = []
   labels = []
   labelsfolds = []
   i = 0
   crossnum = 0
   for crossnum in range(folds):
       Val = []
       Finallabels = []
       for i in range(len(class_indices)):
           data.append(TrainData[i * ratio:ratio * (i + 1)])
           labels.append(TrainLabels[i * ratio:ratio * (i + 1)])
           Val.extend(data[i][int(crossnum * (ratio / folds)):int((crossnum + 1) * (ratio / folds))])
           Finallabels.extend(labels[i][int(crossnum * (ratio / folds)):int((crossnum + 1) * (ratio / folds))])
       Kfolds.append(Val)
       labelsfolds.append(Finallabels)
   return Kfolds, labelsfolds


def tuning_parameters_Hog(datapath, S, class_indices, ratio, num_bins, pixels_per_cells, pixels_per_block, folds, C):
#function that return the errors rate on the test set according to different values of the hog parameters
#param: datapath: address of the data
#param: S: the size of the images
#param: class_indices: the classes
#param: ratio: int that describes the number of images for the train set
#param: num_bins: set of integers that describe the number of oriented gradients
#param: pixels_per_cells: set of integer that describe the Spatial cell size
#param: pixels_per_block: parametr for thr HOG algorithm
#param: folds: int that describes the number of fold for the cross validation
#param: C: set of values for the C parameters in the SVM algorithm
#output: Finalscores: matrix of the errors rate according to the values of the hog parameters
   Finalscores = []
   for nb in num_bins:
       scores=[]
       for ppc in pixels_per_block:
           data, labels, size = GetData(datapath, S, class_indices, ratio)
           trainData, testData, TrainLabels, TestLabels = TrainTestSplit(data, labels, size, class_indices, ratio)
           PrepareData = Prepare(trainData, num_bins, pixels_per_cells, ppc)
           scores.append(tuning_parameters_SVM(TrainLabels, PrepareData, folds, ratio, class_indices, C))
       Finalscores.append(scores)
   return Finalscores


def tuning_parameters_SVM(TrainLabels, TrainData, folds, ratio, class_indices, C):
# function that return the errors rate on the test set according to different values of the SVM parameters
# param: TrainLabels: the train labels
# param: TrainData: the train set
# param: class_indices: the classes
# param: ratio: int that describes the number of images for the train set
# param: folds: int that describes the number of fold for the cross validation
# param: C: set of values for the C parameters in the SVM algorithm
# output: Finalscore: list of the errors rate according to the values of the SVM parameters
   finalScore = []
   Kfolds, labelsfolds = kFoldsPartition(TrainLabels, TrainData, folds, ratio, class_indices)
   for c in C:
       clf = SVC(C=c, kernel='linear', decision_function_shape='ovr', verbose=0, random_state=10, max_iter=1000)
       Score = 0
       for k in range(len(Kfolds)):
           val = Kfolds[k]
           labelval = labelsfolds[k]
           train = Kfolds[:k] + Kfolds[k + 1:]
           labeltrain = labelsfolds[:k] + labelsfolds[k + 1:]
           final_train = []
           final_labels = []
           for x in range(len(train)):
               final_train.extend(train[x])
               final_labels.extend(labeltrain[x])
           clf.fit(final_train, final_labels)
           predict = clf.decision_function(val)
           Results = np.argmax(predict, axis=1)
           Score = Score + Evaluate(labelval, Results)[1]
       finalScore.append(Score / folds)
   return finalScore



def tuning_parameters_HogS(datapath, S, class_indices, ratio, num_bins, pixels_per_cells, pixels_per_block, folds, C, g,BN):
# function that return the errors rate on the test set according to different values of the S parameter
#param: datapath: address of the data
#param: S: set of integers that describe the size of the images
#param: class_indices: the classes
#param: ratio: int that describes the number of images for the train set
#param: num_bins: set of integers that describe the number of oriented gradients
#param: pixels_per_cells: set of integer that describe the Spatial cell size
#param: pixels_per_block: parametr for thr HOG algorithm
#param: folds: int that describes the number of fold for the cross validation
#param: C: set of values for the C parameters in the SVM algorithm
#output: Finalscores: matrix of the errors rate according to the values of the hog parameters
   Finalscores=[]
   for b in BN:
       data, labels, size = GetData(datapath, S, class_indices,ratio)
       trainData, testData, TrainLabels, TestLabels = TrainTestSplit(data, labels, size, class_indices, ratio)
       PrepareData = Prepare(trainData, num_bins, pixels_per_cells, pixels_per_block,b)
       Finalscores.append(tuning_parameters_SVM(TrainLabels, PrepareData, folds, ratio, class_indices, C))

   return Finalscores



def confused_pic(class_indices, y, prediction, decision_matrix,testData):
   wrong_classify = y != prediction
   index_wrong = np.argwhere(wrong_classify == True)  # indices for misclassified example
   index_wrong= index_wrong[:, 0]
   y_wrong = y[wrong_classify]  # belonged class for the misclassified
   prediction_wrong = prediction[wrong_classify]  # the predicted class for the misclassified examples
   wrong_decision_matrix = decision_matrix[wrong_classify, :]  # only the rows that misclassified

   # mapping class to index -> the first class will map to 0 the second to 1 and go on...
   j = 0  # counter
   y_wrong_mapping = np.copy(y_wrong)
   prediction_wrong_mapping = np.copy(prediction_wrong)
   for i in class_indices:
       y_wrong_mapping[y_wrong_mapping == i] = j
       prediction_wrong_mapping[prediction_wrong_mapping == i] = j
       j = j + 1
   y_wrong_mapping = y_wrong_mapping.astype(int)
   prediction_wrong_mapping = prediction_wrong_mapping.astype(int)

   # create vector with the mistake value to the correct class from the predicted class
   mistakes_values = np.asarray([])
   for i in range(y_wrong.shape[0]):  # loop over each row of wrong_decision_matrix, all the misclassified
       mistake = wrong_decision_matrix[i, y_wrong_mapping[i]] - wrong_decision_matrix[i, prediction_wrong_mapping[i]]
       mistakes_values = np.append(mistakes_values, mistake)

   # combine mistake, belonged class, original index to one matrix
   combine = np.asarray([mistakes_values, y_wrong, index_wrong])
   combine = np.transpose(combine)
   # sorting the combined matrix by the mistake
   combine = combine[combine[:, 0].argsort()[::-1]]
   # sorting the combined matrix by the mistake and then by class
   combine = combine[combine[:, 1].argsort()]

   # find 2 largest mistakes
   example_idx = np.asarray([])
   belonged_class = np.asarray([])
   for i in range(combine.shape[0]):
       if example_idx.shape[0] >= 2:
           if belonged_class[-1] == combine[i, 1] and belonged_class[-2] == combine[i, 1]:
               continue
           else:
               example_idx = np.append(example_idx, combine[i, 2])
               belonged_class = np.append(belonged_class, combine[i, 1])
       else:
           example_idx = np.append(example_idx, combine[i, 2])
           belonged_class = np.append(belonged_class, combine[i, 1])

   # show the images
   for i in range(example_idx.shape[0]):
       cv2.imshow('',testData[example_idx.astype(int)][i])
       cv2.waitKey()
       cv2.destroyAllWindows()
       plt.imshow(testData[example_idx.astype(int)][i], cmap = 'gray')
       plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
       plt.show()
   return


def main():
#the program
    parameters= GetDefaultParameters()
    data, labels, size = GetData(parameters['data_path'], parameters['S'],
                                parameters['class_indices'], parameters['ratio'])
    trainData, testData, TrainLabels, TestLabels = TrainTestSplit(data, labels,
                                                                 size, parameters['class_indices'],parameters['ratio'])
    trainSet = Prepare(trainData,  parameters['num_bins'],
                      parameters['pixels_per_cell'], parameters['cells_per_block'],parameters['block_norm'])
    classifier = n_class_SVM_train(trainSet, parameters['class_indices'], parameters['C'],
                      parameters['max_iter'], parameters['ratio'], TrainLabels)
    testSet = Prepare(testData, parameters['num_bins'],
                      parameters['pixels_per_cell'], parameters['cells_per_block'], parameters['block_norm'])
    Scores, ScoreMatrix = n_class_SVM_predict(testSet, parameters['class_indices'], classifier)
    Final, err_rate = Evaluate(TestLabels, Scores)
    print('Confusion matrix')
    print(Final)
    print('%s%f' % ('The err rate for the  classifier is: ', err_rate))
    TestLabels = np.reshape(TestLabels, len(TestLabels))
    testData = np.reshape(testData, (len(testData), parameters['S'], parameters['S']))
    confused_pic(parameters['class_indices'], TestLabels, Scores, ScoreMatrix, testData)
    return


if __name__ == "__main__":
    main()

