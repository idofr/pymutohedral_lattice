pymutohedral_lattice
=============================

A python (numpy based) implementation of the original permutohedral lattice filtering code found in
http://graphics.stanford.edu/papers/permutohedral/


Description
-----------

This numpy based version of the permutohedral lattice gaussian filtering is a simple translation of the cpp CPU code provided by the authors.
Due to the nature of Python is runs slower than the cpp version (ignoring the fact that multiple CUDA implementations are also available out there) and is therefore not recommended for production usage.
Yet it is rather straight forward to understand and very east to use with other Python scripts. It is thus pretty useful for prototyping and educational purposes. 


Scripts
--------------------

* permutohedral_lattice.py includes everything for the filtering.
* main.py contains an example of how to use it. The std parameters are taking from the recommendations of the original paper.

Requirements
------------
We use numpy for most of the computations. skimage and OpenCV3 are used for image reading and showing, which makes them interchangeable.
 You could also replace one for the other as long as the shape of the input images remains the same, namely (row, cols, channels)

Coding Style
------------

We use flake8 in version (3.0.4).