# This sub-repo contains a set of pre-processed UCI datasets

## The unprocessed datasets can be found in [UCI repository](https://archive.ics.uci.edu/ml/index.php)

Every folder has two files-
* A .csv file 
* A .mat file

UCI datasets are generally maintained in .dat format which are harder to format.
I have converted the selected set of data into csv format.

Every csv file has -
* set of features (all the columns except the last one)
* class labels (last column of each dataset)

### _csv files are extremely useful and easy to format. Almost all the major programming languages have libraries to work with csv files_

The structure of the mat file is-
* data
    * train
    * trainLabel
    * test
    * testLabel
    
e.g. Suppose I want to get the data for BreastCancer. Then the steps are:
* dataset = 'BreastCancer';
* data = importdata(strcat('Data/',dataset,'/',dataset,'_data.mat'));
* x = data.train;
* t = data.trainLabel;
* x2 = data.test;
* t2 = data.testLabel;


### _Specifically for MATLAB, I feel that the test-train division is a bit difficult to perform. Instead of using csv for MATLAB, I prefer processing the data and storing them in mat files before use_

The mat files contain data division in 70-30 format (70% training, 30% testing). Validation sets are not provided, I prefer using k-fold cross validation with training data only. So, validation sets are not required at all.

### The dataset description (No. of classes, features, samples) are mentioned in "UCI_Dataset_Description.xlsx" file

Till now, this sub-repo contains following pre-processed UCI datasets:
1. Arrhythmia
2. BreastCancer
3. BreastEW
4. CongressEW
5. Exactly
6. Exactly2
7. Glass
8. HeartEW
9. Hill-valley
10. Horse
11. Ionosphere
12. KrVsKpEW
13. Lymphography
14. Madelon
15. M-of-n
16. Monk1
17. Monk2
18. Monk3
19. PenglungEW
20. Sonar
21. Soybean-small
22. SpectEW
23. Tic-tac-toe
24. Vote
25. Vowel
26. WaveformEW
27. Wine
28. Zoo
