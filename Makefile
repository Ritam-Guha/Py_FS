clean:
	rm -rf build dist

upload:
	twine upload dist/*

run:
	make clean
	python setup.py
	make upload

