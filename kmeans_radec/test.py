"""
Converted from Chis Davis' ipython notebook
"""

from .kmeans_radec import kmeans_sample, KMeans
import numpy

def test():
    from matplotlib.pyplot import draw, figure, show

    # first sample points that do not cross over ra=360
    minra = 50
    maxra = 70
    mindec = -10
    maxdec = 10
    N = 10000
    raoffset = 0
    ncen = 10
    h = 0.1

    fig1 = figure()
    ax1=fig1.add_subplot(111)

    ra, dec = generate_randoms_radec(minra, maxra, mindec, maxdec, N, raoffset)
    ax1.plot(ra, dec, '.', alpha=0.1)
    ax1.set_xlabel('ra')
    ax1.set_ylabel('dec')

    X = numpy.vstack((ra, dec)).T
    km = kmeans_sample(X, ncen)
    plot_centers(km.centers, ax1,
                 x_min=minra, x_max=maxra, y_min=mindec, y_max=maxdec, h=h)
    draw()
    print(km.centers)


    # now sample ones that do cross over the zero line
    # first sample points that do not cross over ra=360
    minra = 350 
    maxra = 370
    mindec = -10
    maxdec = 10
    N = 10000
    raoffset = -180
    ncen = 10
    h = 0.1

    phimin = (minra - 180 + raoffset)
    phimax = (maxra - 180 + raoffset)
    print(phimin, phimax)

    fig2=figure()
    ax2=fig2.add_subplot(111)

    ra, dec = generate_randoms_radec(minra, maxra, mindec, maxdec, N, raoffset)
    ra = numpy.where(ra > 360, ra - 360, ra)
    ax2.plot(ra, dec, '.', alpha=0.2)
    ax2.set_xlabel('ra')
    ax2.set_ylabel('dec')

    X = numpy.vstack((ra, dec)).T
    km = kmeans_sample(X, ncen)
    plot_centers(km.centers, ax2,
                 x_min=0, x_max=360, y_min=mindec, y_max=maxdec, h=10 * h)
    draw()
    print(km.centers)

    # make rotated figure
    fig3 = figure()
    ax3 = fig3.add_subplot(111)

    centers = km.centers.copy()
    centers[:,0] = numpy.where(centers[:,0] > 180, centers[:,0] - 360, centers[:,0])
    ra = numpy.where(ra > 180, ra - 360, ra)
    ax3.plot(ra, dec, '.', alpha=0.2)
    ax3.set_xlabel('ra')
    ax3.set_ylabel('dec')

    plot_centers(centers, ax3,
                 x_min=minra - 360, x_max=maxra - 360, y_min=mindec, y_max=maxdec, h=h)
    draw()
    show()

def generate_randoms_radec(minra, maxra, mindec, maxdec, N, raoffset=0):
    r = 1.0
    # this z is not redshift!
    zmin = r * numpy.sin(numpy.pi * mindec / 180.)
    zmax = r * numpy.sin(numpy.pi * maxdec / 180.)

    phimin = numpy.pi / 180. * (minra - 180 + raoffset)
    phimax = numpy.pi / 180. * (maxra - 180 + raoffset)

    # generate ra and dec
    z_coord = numpy.random.uniform(zmin, zmax, N)  # not redshift!
    phi = numpy.random.uniform(phimin, phimax, N)
    dec_rad = numpy.arcsin(z_coord / r)

    # convert to ra and dec
    random_ra = phi * 180 / numpy.pi + 180 - raoffset
    random_dec = dec_rad * 180 / numpy.pi
    
    return random_ra, random_dec

def plot_centers(centers, ax, x_min=0, x_max=360, y_min=-180, y_max=180, h=1):


    km = KMeans(centers)
    # h is Step size of the mesh. Decrease to increase the quality of the VQ.

    # Plot the decision boundary. For that, we will assign a color to each
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = km.find_nearest(numpy.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    zz = Z.reshape(xx.shape)

    number = numpy.arange(len(centers))
    ax.pcolor(xx, yy, zz)
    for ith, center in enumerate(centers):
        ax.text(center[0], center[1], ith, color='white', fontsize=15)
    #draw()


if __name__=="__main__":
    main()

