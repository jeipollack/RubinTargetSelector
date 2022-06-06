# Gives a representation of the sky coverage of a list of targets
# for some instrument.
#
# An instrument is defined by the shape (a polygon) of its focal plane 
# for RA=0, Dec=0
#   - a json file with a list of (RA, Dec) defining the focal plane polygon 
#     centered at RA=0, Dec=0
# A list of targets is given in the Euclid/SIM format:
#   - json file
#   - list of objects with RA and Dec fields (in degrees)
# Output format:
#   - a rectangular zone defined by RA_min, RA_max, Dec_min, Dec_max, 
#     and number of pixels n_ra, n_dec
#   - all healpix pixels (nside) within a rectangular zone RA_min, ...

import json
import numpy as np
import argparse
import geom
import wn


def read_focal_plane(filename):
    with open(filename) as f:
        vertices = json.load(f)
    polygon = np.array(vertices)
    return polygon


def read_targets(filename):
    with open(filename) as f:
        targets = json.load(f)
        coords = np.array([
            [target['RA'], target['Dec']] 
            for target in targets]
        )
        return coords


def to_slice(string):
    def int_or_none(x):
        if x is not None:
            return int(x)
    items = string.split(':')
    start = items[0] or None
    stop = items[1] or None if len(items) >= 2 else None
    step = items[2] or None if len(items) == 3 else None
    return slice(int_or_none(start), int_or_none(stop), int_or_none(step))


def tuple_of(type):
    def make_tuple(string):
        return tuple(type(x) for x in string.split(','))
    return make_tuple


def make_pixels(extent, size):
    xmin, ymin, xmax, ymax = extent
    nx, ny = size
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xy = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    return xy


def make_polygons(targets, focalplane):
    return [geom.rotate_to(ra, dec, focalplane) for ra, dec in targets]


def close_polygon(polygon):
    """Check the last point in the polygon is identical to the first one, otherwise
    append it. Returns the updated polygon.
    """
    if tuple(polygon[0]) != tuple(polygon[-1]):
        new_polygon = np.zeros((len(polygon)+1, 2))
        new_polygon[:-1, :] = polygon[...]
        new_polygon[-1, :] = polygon[0, :]
    else:
        new_polygon = polygon.copy()
    return new_polygon


def coverage(pixels, polygon):
    return np.int32(wn.wn_multipnpoly(pixels, polygon) != 0)


def coverage_all(pixels, polygons):
    image = coverage(pixels, polygons[0])
    for polygon in polygons[1:]:
        image += coverage(pixels, polygon)
    return image


def make_image(pixels, coverage, extent, size):
    xmin, ymin, xmax, ymax = extent
    nx, ny = size
    dx = (xmax-xmin)/(nx-1)
    dy = (ymax-ymin)/(ny-1)
    x = np.linspace(xmin-dx/2, xmax+dx/2, nx+1)
    y = np.linspace(ymin-dy/2, ymax+dy/2, ny+1)
    image = np.reshape(coverage, (nx, ny)).T
    return x, y, image


def make_histogram(pixels, coverage, polygon=None, **kwds):
    if polygon is not None:
        keep = coverage(pixels, polygon)
    return np.histogram(coverage[keep], **kwds)


def display_image(x, y, image, title=None, clim=None):
    from matplotlib import pyplot as plt
    plt.figure()
    plt.pcolormesh(x, y, np.ma.masked_equal(image, 0))
    plt.grid()
    plt.colorbar()
    if clim is not None:
        cmin, cmax = clim
        plt.clim(cmin, cmax)
    if title is not None:
        plt.title(title)
    plt.axis('equal')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Sky coverage of targets.')
    parser.add_argument(
        '--targets',
        required=True,
        type=read_targets,
        help='name of json file containing targets'
    )
    parser.add_argument(
        '--slice',
        type=to_slice,
        default=slice(None),
        help='Slice of targets in the form [start]:[stop]:[step]'
    )
    parser.add_argument(
        '--focalplane',
        required=True,
        type=read_focal_plane,
        help='name of json file containing vertices of polygon enclosing '
             'the focal plane'
    )
    parser.add_argument(
        '--output',
        help='name of output file'
    )
    parser.add_argument(
        '--extent',
        type=tuple_of(float),
        help='The extension of output image: xmin,ymin,xmax,ymax'
    )
    parser.add_argument(
        '--size',
        type=tuple_of(int),
        help='The size of the output image: nx,ny'
    )
    parser.add_argument(
        '--display',
        action='store_true',
        help='Display the output image'
    )
    parser.add_argument(
        '--title',
        help='Title of the displayed image'
    )
    parser.add_argument(
        '--clim',
        type=tuple_of(float),
        help='The color limits of the colorbar'
    )
    parser.add_argument(
        '--ok',
        type=tuple_of(float),
        help='Interval of ok values'
    )

    args = parser.parse_args()
    focalplane = close_polygon(args.focalplane)
    targets = args.targets[args.slice]
    print(f'Focal plane polygon: {focalplane}')
    print(f'Number of targets: {len(targets)}')

    xmin, ymin, xmax, ymax = args.extent
    xmin = xmin - 360 if xmin > 180 else xmin
    xmax = xmax - 360 if xmax > 180 else xmax
    extent = xmin, ymin, xmax, ymax
    pixels = make_pixels(extent, args.size)
    polygons = make_polygons(targets, focalplane)
    cover = coverage_all(pixels, polygons)
    x, y, image = make_image(pixels, cover, extent, args.size)

    if args.output is not None:
        #print(f'Writing image to {args.output}')
        np.savez(args.output, size=args.size, extent=extent,
                pixels=pixels, coverage=cover, 
                x=x, y=y, image=image,
                polygons=polygons,
                targets=targets)
    
    if args.ok is not None:
        minok, maxok = args.ok
        image[(image < minok) & (image != 0)] = minok
        image[image > maxok] = maxok

    if args.display:
        display_image(x, y, image, title=args.title, clim=args.clim)

if __name__ == '__main__':
    main()
