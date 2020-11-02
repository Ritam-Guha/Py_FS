# Py_FS: A Python Package for Feature Selection
Py_FS is a toolbox developed with complete focus on Feature Selection (FS) using Python as the underlying programming language. It comes with capabilities like nature-inspired evolutionary feature selection algorithms, filter methods and simple evaulation metrics to help with easy applications and comparisons among different feature selection algorithms over different datasets. It is still in the development phase. We wish to extend this package further to contain more extensive set of feature selection procedures and corresponding utilities.

<p align="center">  
  <img src="https://github.com/Ritam-Guha/Py_FS/blob/master/Images/logo.jpg" height="300" width="300">
</p><br>

## Installation

Please install the required utilities for the package by running this piece of code:

    pip3 install -r requirements.txt

The package is publicly avaliable at **PYPI: Python Package Index**.
Anybody willing to use the package can install it by simply running:
    
    pip3 install Py_FS



## Structure

The current structure of the package is mentioned below. Depending on the level of the function intended to call, it should be imported using the period(.) hierarchy.

### Py_FS

* [filter/](./Py_FS/filter)
  * [MI.py](./Py_FS/filter/MI.py)
  * [PCC.py](./Py_FS/filter/PCC.py)
  * [Relief.py](./Py_FS/filter/Relief.py)
  * [SCC.py](./Py_FS/filter/SCC.py)
* [wrapper/](./Py_FS/wrapper)
  * [nature_inspired/](./Py_FS/wrapper/nature_inspired)
    * [BBA.py](./Py_FS/wrapper/nature_inspired/BBA.py)
    * [CS.py](./Py_FS/wrapper/nature_inspired/CS.py)
    * [EO.py](./Py_FS/wrapper/nature_inspired/EO.py)
    * [GA.py](./Py_FS/wrapper/nature_inspired/GA.py)
    * [GSA.py](./Py_FS/wrapper/nature_inspired/GSA.py)
    * [GWO.py](./Py_FS/wrapper/nature_inspired/GWO.py)
    * [HS.py](./Py_FS/wrapper/nature_inspired/HS.py)
    * [MF.py](./Py_FS/wrapper/nature_inspired/MF.py)
    * [PSO.py](./Py_FS/wrapper/nature_inspired/PSO.py)
    * [RDA.py](./Py_FS/wrapper/nature_inspired/RDA.py)
    * [SCA.py](./Py_FS/wrapper/nature_inspired/SCA.py)
    * [WOA.py](./Py_FS/wrapper/nature_inspired/WOA.py)
* [evaluation.py](./Py_FS/evaluation.py)


For example, if someone wants to use GA, it should be imported using the following statement:

    import Py_FS.wrapper.nature_inspired.GA

There are mainly three utilities in the current version of the package. The next part discusses these
three parts in detail:

## Quick User Guide
For a quick demonstration of the process of using Py_FS, please proceed to this Colab link: [Py_FS: Demonstration](https://colab.research.google.com/drive/1PafNTmVgWv9Qz6j7bI41XqPT6CCCIb1T?usp=sharing).

## References
This toolbox has been developed by a team of students from [Computer Science and Engineering department, Jadavpur University](http://www.jaduniv.edu.in/view_department.php?deptid=59) supervised by [Prof. Ram Sarkar](http://www.jaduniv.edu.in/profile.php?uid=686). This team has participated in many research activities related to engineering optimization, feature selection and image processing. We request the users of Py_FS to cite the relevant articles from our group. It will mean a lot to us. The articles produced by this team are mentioned below:

### Wrappers

#### GA

* Guha, R., Ghosh, M., Kapri, S., Shaw, S., Mutsuddi, S., Bhateja, V., & Sarkar, R. (2019). Deluge based Genetic Algorithm for feature selection. Evolutionary intelligence, 1-11.

* Ghosh, M., Guha, R., Mondal, R., Singh, P. K., Sarkar, R., & Nasipuri, M. (2018). Feature selection using histogram-based multi-objective GA for handwritten Devanagari numeral recognition. In Intelligent engineering informatics (pp. 471-479). Springer, Singapore.

* Guha, R., Ghosh, M., Singh, P. K., Sarkar, R., & Nasipuri, M. (2019). M-HMOGA: a new multi-objective feature selection algorithm for handwritten numeral classification. Journal of Intelligent Systems, 29(1), 1453-1467.

* Ghosh, M., Guha, R., Singh, P. K., Bhateja, V., & Sarkar, R. (2019). A histogram based fuzzy ensemble technique for feature selection. Evolutionary Intelligence, 12(4), 713-724.

* Ghosh, M., Guha, R., Alam, I., Lohariwal, P., Jalan, D., & Sarkar, R. (2019). Binary Genetic Swarm Optimization: A Combination of GA and PSO for Feature Selection. Journal of Intelligent Systems, 29(1), 1598-1610.

* Guha, R., Khan, A.H., Singh, P.K. et al. CGA: a new feature selection model for visual human action recognition. Neural Comput & Applic (2020). https://doi.org/10.1007/s00521-020-05297-5

#### GSA

* Guha, R., Ghosh, M., Chakrabarti, A., Sarkar, R., & Mirjalili, S. (2020). Introducing clustering based population in Binary Gravitational Search Algorithm for Feature Selection. Applied Soft Computing, 106341.

#### GWO

* Dhargupta, S., Ghosh, M., Mirjalili, S., & Sarkar, R. (2020). Selective opposition based grey wolf optimization. Expert Systems with Applications, 113389.


#### MA

* T. Bhattacharyya, B. Chatterjee, P. K. Singh, J. H. Yoon, Z. W. Geem and R. Sarkar, "Mayfly in Harmony: A New Hybrid Meta-heuristic Feature Selection Algorithm," in IEEE Access, doi: 10.1109/ACCESS.2020.3031718

#### PSO

* Ghosh, M., Guha, R., Singh, P. K., Bhateja, V., & Sarkar, R. (2019). A histogram based fuzzy ensemble technique for feature selection. Evolutionary Intelligence, 12(4), 713-724.

* Ghosh, M., Guha, R., Alam, I., Lohariwal, P., Jalan, D., & Sarkar, R. (2019). Binary Genetic Swarm Optimization: A Combination of GA and PSO for Feature Selection. Journal of Intelligent Systems, 29(1), 1598-1610.

#### WOA

* Guha, R., Ghosh, M., Mutsuddi, S. et al. Embedded chaotic whale survival algorithm for filter–wrapper feature selection. Soft Comput 24, 12821–12843 (2020). https://doi.org/10.1007/s00500-020-05183-1.

### Filters

#### PCC

* Guha, R., Ghosh, K. K., Bhowmik, S., & Sarkar, R. (2020, February). Mutually Informed Correlation Coefficient (MICC)-a New Filter Based Feature Selection Method. In 2020 IEEE Calcutta Conference (CALCON) (pp. 54-58). IEEE.

#### MI

* Guha, R., Ghosh, K. K., Bhowmik, S., & Sarkar, R. (2020, February). Mutually Informed Correlation Coefficient (MICC)-a New Filter Based Feature Selection Method. In 2020 IEEE Calcutta Conference (CALCON) (pp. 54-58). IEEE.  

## 1. Wrapper-based Nature-inspired Feature Selection
Wrapper-based Nature-inspired methods are very popular feature selection approaches due to their efficiency and simplicity. These methods progress by introducing random set of candidate solutions (agents which are natural elements like particles, whales, bats etc.) and improving these solutions gradually by using guidance mechanisms of fitter agents. In order to calculate the fitness of the candidate solutions, wrappers require some learning algorithm (like classifiers) to calculate the worth of a solution at every iteration. This makes wrapper methods extremely reliable but computationally expensive as well.

Py_FS currently supports the following 12 wrapper-based FS methods:
* Binary Bat Algorithm (BBA)
* Cuckoo Search Algorithm (CS)
* Equilibrium Optimizer (EO)
* Genetic Algorithm (GA)
* Gravitational Search Algorithm (GSA)
* Grey Wolf Optimizer (GWO)
* Harmony Search (HS)
* Mayfly Algorithm (MA)
* Particle Swarm Optimization (PSO)
* Red Deer Algorithm (RDA)
* Sine Cosine Algorithm (SCA)
* Whale Optimization Algorithm (WOA)

These wrapper approaches can be imported in your code using the following statements:
    
    import Py_FS.wrapper.nature_inspired.BBA
    import Py_FS.wrapper.nature_inspired.CS
    import Py_FS.wrapper.nature_inspired.EO
    import Py_FS.wrapper.nature_inspired.GA
    import Py_FS.wrapper.nature_inspired.GSA
    import Py_FS.wrapper.nature_inspired.GWO
    import Py_FS.wrapper.nature_inspired.HS
    import Py_FS.wrapper.nature_inspired.MA
    import Py_FS.wrapper.nature_inspired.PSO
    import Py_FS.wrapper.nature_inspired.RDA
    import Py_FS.wrapper.nature_inspired.SCA
    import Py_FS.wrapper.nature_inspired.WOA

## 2. Filter-based Feature Selection
Filter methods do not use any intermediate learning algorithm to verify the strength of the generated solutions. Instead, they use statistical measures to identify the importance of different features in the context. So, finally every feature gets a rank according to their relevance in the dataset. The top features can then be used for classification. 

Py_FS currently supports the following 4 filter-based FS methods:
* Pearson Correlation Coefficient (PCC)
* Spearman Correlation Coefficient (SCC)
* Relief
* Mutual Information (MI)

These filter approaches can be imported in your code using the following statements:
    
    import Py_FS.filter.PCC
    import Py_FS.filter.SCC
    import Py_FS.filter.Relief
    import Py_FS.filter.MI

## 3. Evaluation Metrics
The package comes with tools to evaluate features before or after FS. This helps to easily compare and analyze 
performances of different FS procedures. 

Py_FS currently supports the following evaluation metrics:
* classification accuracy
* average recall
* average precision
* average f1 score
* confusion matrix
* confusion graph

The evaulation capabilities can be imported in your code using the following statement:
    
    from Py_FS.evaluation import evaluate
