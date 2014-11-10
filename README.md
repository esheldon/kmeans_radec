kmeans_radec
============

K means algorithm on the unit sphere

examples
--------

```python
import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample

# first lets try an easy example, letting the code generate
# an initial set of guesses for the centers


# find 40 centers
ncen = 40

# the data X has shape [Npoints, ra, dec]
# run no more than 100 iterations. Will stop if tolerance is met
km = kmeans_sample(X, ncen, maxiter=100, tol=1.0e-5)

# we got back a KMeans object:

# the centers found by the algorithm
print("found centers:",km.centers)

# did we converge?
print("converged?",self.converged)

# the index to the nearest center for each point in X
print("labels size:",km.labels.size)

# how many in each label? Should be fairly uniform
print("cluster sizes:", numpy.bincount(km.labels))

# the distance to each center [Npoints, Ncen]
print("shape of distances:",km.distances.shape)

#
# maybe we want more control, let's use a KMeans object
# start with our own guess
#

km=KMeans(cen_guess)

# run the algorithem
km.run(X, maxiter=100)

# did it converge?
if km.converged:
    print("k means converged to :",km.centers)

    # find nearest centers to a *different* set of points
    labels=km.find_nearest(X2)

    # equivalent to 
    labels=kmeans_radec.find_nearest(X2, km.centers)

else:

    # no convergence, we could just run more
    #     km.run(maxiter=maxiter)
    # or we could try a different set of center guesses...

    print("k means did not converge, trying different guesses")

    km.set_centers(cen_guess2)
    km.run(X, maxiter=100)
    # ... etc.
```

dependencies
------------
numpy
