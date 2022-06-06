import skycoverage
import matplotlib.pyplot as plt
import numpy as np

def get_layers(extent,size,targets,focalplane):
    pixels = skycoverage.make_pixels(extent,size)
    polygons = skycoverage.make_polygons(targets,focalplane)

    cover = skycoverage.coverage_all(pixels,polygons)
    
    X, Y, image = skycoverage.make_image(pixels, cover, extent, size)
#    x, y, image = data['x']+360,data['y'],data['image']
    return X,Y, image
