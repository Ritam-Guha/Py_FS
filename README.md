# Py_FS: A Python Package for Feature Selection

Py_FS is a toolbox developed with complete focus on Feature Selection (FS) using Python as the underlying programming language. It comes with capabilities like nature-inspired evolutionary feature selection algorithms, filter methods and simple evaulation metrics to help with easy applications and comparisons among different feature selection algorithms over different datasets. It is still in the development phase. We wish to extend this package further to contain more extensive set of feature selection procedures and corresponding utilities.

## Installation

The package is publicly avaliable at PYPI: Python Package Index.
Anybody willing to use the package can install it by simply calling:
    
    pip install Py_FS

## Structure

The current structure of the package is mentioned below. Depending on the level of the function intended to call, the package should be imported using the period(.) hierarchy.

Py_FS/
┣ filter/
┃ ┣ PCC.py
┣ wrapper/
┃ ┣ nature_inspired/
┃ ┃ ┣ GA.py
┃ ┃ ┣ GWO.py
┃ ┃ ┣ HS.py
┃ ┃ ┣ MF.py
┃ ┃ ┣ PSO.py
┃ ┃ ┣ RDA.py
┃ ┃ ┣ WOA.py
┗ evaluation.py

For example, if someone wants to use GA, he needs to import GA using the following statement:

    import Py_FS.wrapper.GA