# set the directory path
import os, sys
import os.path as path
abs_path_pkg =  path.abspath(path.join(__file__ ,"../../../"))
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, abs_path_pkg)

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVM
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import matplotlib.pyplot as plt

class Metric():
    # class for defining the evaluation metrics
    def __init__(self, train_X, test_X, train_Y, test_Y, agent, classifier, save_conf_mat, averaging):
        
        self.agent = agent
        if self.agent is None:
            self.agent = np.ones(train_X.shape[1])

        cols = np.flatnonzero(self.agent)  
        if(cols.shape[0] == 0):
            print('[Error!] There are 0 features in the agent......')   
            exit(1) 

        # store the train and test features and labels
        self.train_X = train_X[:, cols]
        self.test_X = test_X[:, cols]
        self.train_Y = train_Y
        self.test_Y = test_Y

        # set the classifier type
        self.classifier = classifier

        # get the unique labels
        self.labels = np.unique(train_Y)

        # select the averaging procedure
        if (len(self.labels) == 2):
            self.averaging = "binary"
        else:
            self.averaging = averaging
        
        # setup the classifier
        if(self.classifier.lower() == 'knn'):
            self.clf = KNN()
        elif(self.classifier.lower() == 'rf'):
            self.clf = RF()
        elif(self.classifier.lower() == 'svm'):
            self.clf = SVM()
        else:
            self.clf = None
            print('\n[Error!] We don\'t currently support {} classifier...\n'.format(classifier))
            exit(1)
            
        # call the member functions 
        self.predictions = self.classify()
        self.accuracy = self.compute_accuracy()
        self.precision = self.compute_precision()
        self.recall = self.compute_recall()
        self.f1_score = self.compute_f1()
        self.confusion_matrix = self.compute_confusion_matrix()
        self.plot_confusion_matrix(save_conf_mat)
        

    def classify(self):
        # function to predict the labels for the test samples 
        train_data = self.train_X
        train_label = self.train_Y
        test_data = self.test_X

        self.clf.fit(train_data,train_label)
        predictions = self.clf.predict(test_data)

        return predictions


    def compute_accuracy(self):
        # function to compute the classification accuracy
        total_count = self.test_Y.shape[0]
        correct_count = np.sum(self.predictions == self.test_Y)
        acc = correct_count/total_count

        return acc
        

    def compute_precision(self):
        # function to compute the average precision
        precision = None

        if(len(self.labels) == 2):
            # binary labels require pos_label
            precision = {}
            precision[self.labels[0]] = precision_score(self.test_Y, self.predictions, pos_label=self.labels[0] ,average=self.averaging)
            precision[self.labels[1]] = precision_score(self.test_Y, self.predictions, pos_label=self.labels[1] ,average=self.averaging)

        else:
            precision = precision_score(self.test_Y, self.predictions, average=self.averaging)
        
        return precision

    def compute_recall(self):
        # function to compute the average recall
        recall = None

        if(len(self.labels) == 2):
            # binary labels require pos_label
            recall = {}
            recall[self.labels[0]] = recall_score(self.test_Y, self.predictions, pos_label=self.labels[0] ,average=self.averaging)
            recall[self.labels[1]] = recall_score(self.test_Y, self.predictions, pos_label=self.labels[1] ,average=self.averaging)

        else:
            recall = recall_score(self.test_Y, self.predictions, average=self.averaging)
        
        return recall

    def compute_f1(self):
        # function to compute the f1 score
        f1 = None

        if(len(self.labels) == 2):
            # binary labels require pos_label
            f1 = {}
            f1[self.labels[0]] = f1_score(self.test_Y, self.predictions, pos_label=self.labels[0] ,average=self.averaging)
            f1[self.labels[1]] = f1_score(self.test_Y, self.predictions, pos_label=self.labels[1] ,average=self.averaging)

        else:
            f1 = f1_score(self.test_Y, self.predictions, average=self.averaging)
        
        return f1

    def compute_confusion_matrix(self):
        # function to compute the confusion matrix
        return confusion_matrix(self.test_Y, self.predictions)

    def plot_confusion_matrix(self, save_conf_mat=False):
        # function to plot the confusion matrix
        ConfusionMatrixDisplay.from_estimator(self.clf, self.test_X, self.test_Y)
        if(save_conf_mat):
            plt.savefig('confusion_matrix.jpg')
        plt.title('Confusion Matrix')
        plt.show()