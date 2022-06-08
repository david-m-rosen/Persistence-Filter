# Persistence Filter

The **Persistence Filter** is an efficient Bayesian algorithm for estimating a belief over the continued existence of environmental features, a common problem arising in robotic mapping in semi-static (i.e., *changing*) environments.

We are making this software freely available in the hope that it will be useful to others.  If you use the Persistence Filter in your own work, please cite our [paper](https://david-m-rosen.github.io/publication/persistencefilter-icra/PersistenceFilter-ICRA.pdf): 

```
@inproceedings{Rosen2016Towards,
title = {Towards Lifelong Feature-Based Mapping in Semi-Static Environments},
author = {Rosen, D.M. and Mason, J. and Leonard, J.J.},
booktitle = {{IEEE} Intl. Conf. on Robotics and Automation (ICRA)},
pages = {1063--1070},
address = {Stockholm, Sweden},
month = may,
year = 2016,
}
```

### Compiling on MacOS

In order to get the persistence filter code to compile on MacOS I had to make a few modifications to the `CMakeLists.txt` file. It is worth taking a look at the Apple-specific content running in the updated `CMakeLists.txt` file, as you may have to modify some paths to make things work properly on your machine.

Before doing anything, I had to install a few packages from `brew`:
```
# Install libgsl (GNU Scientific Library)
brew install gsl

# Install Boost Python3
brew install boost-python3
```

Now you should be ready to try to compile. I prefixed my CMake command below with the path to the Boost library directory so that CMake could find it. Maybe there is a better way, though.

From the project root directory:
```
mkdir build
cd build
BOOST_LIBRARYDIR=/usr/local/Cellar/boost-python3/1.78.0/lib cmake ..
make
```

Now you can test the code by running:
```
./persistence_filter_test
```

### Copyright and License

The C++ and Python implementations of the Persistence Filter contained herein are copyright (C) 2016 by David M. Rosen, and are distributed under the terms of the GNU General Public License (GPL) version 3 (or later).  Please see the file LICENSE for more information.

Contact: drosen2000@gmail.com
