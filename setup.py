import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='Py_FS',  
     version='0.0.6.1',
     author="Ritam Guha",
     author_email="ritamguha16@gmail.com",
     description="A Python Package for Feature Selection",
     long_description=long_description,
    long_description_content_type="text/markdown",
     url="https://github.com/Ritam-Guha/Py_FS",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )