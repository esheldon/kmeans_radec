"""
k means on the sphere

Adapted from this stack overflow answer

http://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikit-learn-k-means

"""
from __future__ import print_function
from __future__ import division

import random
import numpy

_TOL_DEF=1.0e-5
_MAXITER_DEF=100
_VERBOSE_DEF=1

class KMeans(object):
    """
    A class to perform K-means on the input ra,dec using spherical distances

    parameters
    ----------
    centers_guess: array
        [ncen, ra, dec] starting guesses.  Can reset later with set_centers()
    tol: float, optional
        The relative change in the average distance to
        centers, signifies convergence
    verbose: int, optional
        How verbose.  0 silent, 1 minimal starting info, 2 prints running
        distances

    attributes after running
    ------------------------
    .converged: bool
        True if converged
    .centers: array
        the found centers
    .labels: array
        [N,ra,dec] array
    .distances: array
        Distance from each point to each center
    .X: array
        The data that was processed

    example
    -------
    import kmeans_radec
    from kmeans_radec import KMeans

    cen_guess=numpy.zeros(ncen, 2)
    cen_guess[:,0] = ra_guesses
    cen_guess[:,1] = dec_guesses
    km=KMeans(cen_guess)
    km.run(X, maxiter=100)

    # did it converge?
    if not km.converged:
        # did not converge.  This might be ok, but if we want
        # to run more we can
        km.run(maxiter=maxiter)

        # or we could try a different set of center guesses...
        km.set_centers(cen_guess2)
        km.run(X, maxiter=100)

    # results are saved in attributes
    print(km.centers, km.labels, km.distances)
    print("copy of centers:",km.get_centers())

    # once we have our centers, we can identify to which cluster 
    # a *different* set of points belong.  This could be a set
    # of random points we want to associate with the same regions

    labels=km.find_nearest(X2)

    # you can save the centers and load them into a KMeans
    # object at a later time
    km=KMeans(centers)
    labels=km.find_nearest(X)
    """
    def __init__(self, centers,
                 tol=_TOL_DEF,
                 verbose=_VERBOSE_DEF):

        self.set_centers(centers)

        self.tol=float(tol)
        self.verbose=verbose

    def run(self, X, maxiter=_MAXITER_DEF):
        """
        run k means, either until convergence is reached or the indicated
        number of iterations are performed

        parameters
        ----------
        X: array
            [N, ra, dec] array
        maxiter: int, optional
            Max number of iterations to run.
        """
        centers=self.get_centers()
        _check_dims(X, self.centers)

        N, dim = X.shape
        ncen, cdim = centers.shape

        if self.verbose:
            tup=(X.shape, centers.shape, self.tol, maxiter)
            print("X %s  centers %s  tol=%.2g  maxiter=%d" % tup)

        self.converged=False
        allx = numpy.arange(N)
        prevdist = 0
        for jiter in xrange( 1, maxiter+1 ):

            D = cdist_radec(X, centers)  # npoints x ncenters

            labels = D.argmin(axis=1)  # X -> nearest centre

            distances = D[allx,labels]
            avdist = distances.mean()  # median ?
            if self.verbose >= 2:
                print("    av |X - nearest centre| = %.4g" % avdist)

            self.converged = (1 - self.tol) * prevdist <= avdist <= prevdist
            if self.converged:
                break

            if  jiter==maxiter:
                break

            prevdist = avdist
            for jc in range(ncen):  # (1 pass in C)
                c, = numpy.where( labels == jc )
                if len(c) > 0:
                    centers[jc] = X[c].mean( axis=0 )

        if self.verbose:
            print(jiter,"iterations  cluster "
                  "sizes:", numpy.bincount(labels))
        if self.verbose >= 2:
            self._print_info()

        self.X=X
        self.centers=centers
        self.labels=labels
        self.distances=distances


    def set_centers(self, centers):
        """
        set starting centers

        parameters
        ----------
        centers: array
            [Ncen] array of centers
        """
        centers=numpy.asanyarray(centers)

        # we won't change this
        self.centers_guess=centers.copy()

        # this will evolve during the run
        self.centers=centers.copy()

    def get_centers(self):
        """
        get a copy of the centers
        """

        centers=self.centers
        if centers is None:
            raise ValueError("you must set centers first")

        return centers.copy()

    def find_nearest(self, X):
        """
        find the nearest centers to the input points
        """
        return find_nearest(X, self.centers)

    def _print_info(self):
        ncen=self.centers.size
        r50 = numpy.zeros(ncen)
        r90 = numpy.zeros(ncen)

        distances=self.distances
        labels=self.labels

        for j in range(ncen):
            dist = distances[ labels == j ]
            if len(dist) > 0:
                r50[j], r90[j] = numpy.percentile( dist, (50, 90) )
        print("kmeans: cluster 50 % radius", r50.astype(int))
        print("kmeans: cluster 90 % radius", r90.astype(int))
            # scale L1 / dim, L2 / sqrt(dim) ?

def kmeans(X, centers_guess,
           tol=_TOL_DEF,
           maxiter=_MAXITER_DEF,
           verbose=_VERBOSE_DEF):
    """
    perform kmeans on the input ra,dec using spherical distances

    parameters
    ----------
    X: array
        [N, ra, dec] array
    centers_guess: array
        [ncen, ra, dec] array.  The center guesses.
    tol: float, optional
        The relative change in the average distance to
        centers, signifies convergence
    verbose: int, optional
        How verbose.  0 silent, 1 minimal starting info, 2 prints running
        distances

    returns
    -------
    A KMeans object, with attributes .centers, .labels, .distances etc.

    .converged: bool
        True if converged
    .centers: array
        The array of centers, [ncen, ra, dec]
    .labels: array
        The index of the center closest to each input point [N]
    .distances: array
        The distance to the closest center for each poit [N]
    """

    km=KMeans(centers_guess, tol=tol, verbose=verbose)
    km.run(X, maxiter=maxiter)
    return km

def kmeans_sample(X, ncen, nsample=None, maxiter=_MAXITER_DEF, **kw ):
    """
    2-pass kmeans, fast for large N

    - kmeans a smaller random sample from X
    - take starting guesses for the centers from a random sample
      of the input points
    - full kmeans, starting from the centers from pass 1

    parameters
    ----------
    X: array
        [N, ra, dec] array
    ncen: int
        Number of centers
    nsample: int, optional
        Number of samples to use on first pass, default 
        max( 2*sqrt(N), 10*ncen )
    tol: float, optional
        The relative change in the average distance to
        centers, signifies convergence
    verbose: int, optional
        How verbose.  0 silent, 1 minimal starting info, 2 prints running
        distances

    returns
    -------
    A KMeans object, with attributes .centers, .labels, .distances etc.

    .converged: bool
        True if converged
    .centers: array
        The array of centers, [ncen, ra, dec]
    .labels: array
        The index of the center closest to each input point [N]
    .distances: array
        The distance to the closest center for each poit [N]
    """

    N, dim = X.shape
    if nsample is None:
        nsample = max( 2*numpy.sqrt(N), 10*ncen )

    # smaller random sample to start with
    Xsample = random_sample( X, int(nsample) )

    # choose random sample as centers
    pass1centers = random_sample( X, int(ncen) )

    km=KMeans(pass1centers, **kw)
    km.run(Xsample, maxiter=maxiter)

    # now a full run with these centers
    sample_centers = km.get_centers()

    km=KMeans(sample_centers, **kw)
    km.run(X, maxiter=maxiter)
    
    return km

_PIOVER2=numpy.pi*0.5
def cdist_radec(a1, a2):
    """
    use broadcasting to get all distance pairs

    a represents [N,ra,dec]
    """
    from numpy import cos, sin, arccos, newaxis, deg2rad

    ra1=a1[:,0]
    dec1=a1[:,1]
    ra2=a2[:,0]
    dec2=a2[:,1]

    ra1=ra1[:,newaxis]
    dec1=dec1[:,newaxis]

    phi1 = deg2rad(ra1)
    theta1 = _PIOVER2 - deg2rad(dec1)
    phi2 = deg2rad(ra2)
    theta2 = _PIOVER2 - deg2rad(dec2)

    sintheta = sin(theta1)
    x1 = sintheta * cos(phi1)
    y1 = sintheta * sin(phi1)
    z1 = cos(theta1)

    sintheta = sin(theta2)
    x2 = sintheta * cos(phi2)
    y2 = sintheta * sin(phi2)
    z2 = cos(theta2)

    costheta = x1*x2 + y1*y2 + z1*z2

    costheta=numpy.clip(costheta,-1.0,1.0)
    theta = arccos(costheta)
    return theta


def random_sample( X, n ):
    """
    random.sample of the rows of X
    """
    sampleix = random.sample( xrange( X.shape[0] ), int(n) )
    return X[sampleix]

def find_nearest( X, centers):
    """
    find the nearest center for each input point

    parameters
    ----------
    X: array
        [N,ra,dec] points
    centers: array
        [ncen,ra,dec] center points

    returns
    -------
    labels: array
        The index of the nearest center for each input point
    """
    _check_dims(X, centers)
    D = cdist_radec( X, centers)  # |X| x |centers|
    return D.argmin(axis=1)


def _check_dims(X, centers):
    """
    check the dims are compatible
    """
    N, dim = X.shape
    ncen, cdim = centers.shape
    if dim != cdim:
        tup=(X.shape, centers.shape )
        raise ValueError("X %s and centers %s must have the same "
                         "number of columns" % tup)


