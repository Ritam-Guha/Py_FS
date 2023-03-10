python setup.py sdist bdist_wheel
twine upload dist/*

git commit -m "Updates"
git remote set-url origin https://github.com/Ritam-Guha/Py_FS.git
git push
git remote set-url origin https://github.com/CMATER-JUCS/Py_FS.git
git push