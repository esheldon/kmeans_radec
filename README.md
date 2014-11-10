kmeans_radec
============

K means algorithm on the unit sphere

examples
--------

```python
# In the following, the data array X has shape [Npoints, ra, dec]

import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample

# first lets try an easy example, letting the code generate
# an initial set of guesses for the centers by sampling
# randomly from the data set

# run no more than 100 iterations. Will stop if tolerance is met
ncen = 40
km = kmeans_sample(X, ncen, maxiter=100, tol=1.0e-5)

# we got back a KMeans object:

# the centers found by the algorithm
print("found centers:",km.centers)

# did we converge?
print("converged?",km.converged)

# labels are the index to the nearest center for each point in X
print("labels size:",km.labels.size)

# how many in each cluster? Should be fairly uniform
print("cluster sizes:", numpy.bincount(km.labels))

# the distance to each center [Npoints, Ncen]
print("shape of distances:",km.distances.shape)

#
# maybe we want more control, let's use a KMeans object
# start with our own guess
#

cen_guess=numpy.zeros( (ncen, 2) )
cen_guess[:,0] = ra_guesses
cen_guess[:,1] = dec_guesses
km=KMeans(cen_guess)

# run the algorithem
km.run(X, maxiter=100)

# did it converge?
if not km.converged:
    # did not converge.  This might be ok, but if we want
    # to run more we can
    km.run(X, maxiter=maxiter)

    # or we could try a different set of center guesses...
    km.set_centers(cen_guess2)
    km.run(X, maxiter=100)

# once we have our centers, we can identify to which cluster 
# a *different* set of points belong.  This could be a set
# of random points we want to associate with the same regions

labels=km.find_nearest(X2)

# you can save the centers and load them into a KMeans
# object later
km=KMeans(centers)
labels=km.find_nearest(X)

# the above is equivalent to the simple function call
labels=kmeans_radec.find_nearest(X, centers)
```

installation
------------
```
git clone https://github.com/esheldon/kmeans_radec.git
cd kmeans_radec
python setup.py install

# or in a user-defined prefix
python setup.py install --prefix=/some/path
```

dependencies
------------
numpy
