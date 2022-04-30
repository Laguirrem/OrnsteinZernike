# OrnsteinZernike
Efficient solver of the Ornstein Zernike equation

This project builds a general solver of the Ornstein-Zernike equation based on Newton's method and an efficient linear solver for sparse systems. The core of the solver stands in the OrnsteinZernike class of the OZ_class.py file. For a cleaner code, the auxiliary functions used in this class are defined in the AuxFunctions.py file, specifically, the Fourier transforms.

An implementation of the class can be found in the Pair_Distribution_Microgels_Basic.ipynb file, where the solver is applied to an ionic three component system.

