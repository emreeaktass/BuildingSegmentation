from skimage import io
import numpy as np
import cv2
from osgeo import gdal


def read(path, patch_size, type, transpose, normilize, prun, cdtpye_uint8):
    if 'tiff' in path:
        raster_data = gdal.Open(path)
        cols = raster_data.RasterXSize
        rows = raster_data.RasterYSize
        data = raster_data.ReadAsArray(0, 0, cols, rows)
    else:
        raster_data = gdal.Open(path)
        cols = raster_data.RasterXSize
        rows = raster_data.RasterYSize
        data = raster_data.ReadAsArray(0, 0, cols, rows)
    del raster_data

    if type == 'mask':
        data[data > 0] = 1

    if transpose:
        data = data.transpose((1, 2, 0))

    if prun:
        data[data > 100] = 0
        data[data < 0] = 0

    data_adjusted = resize_with_cut(data, rows, cols, patch_size, type)
    data = []
    if normilize:
        data_adjusted = cv2.normalize(data_adjusted, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    if cdtpye_uint8:
        data_adjusted = data_adjusted.astype(np.uint8)
    return data_adjusted


def write(path, name, type, data):
    np.save(path + '/' + type + '/' + str(name), data)



def resize_with_cut(data, row, col, patch_size, type):
    overageX = col % patch_size
    overageY = row % patch_size
    if type == 'mask' or type == 'height':
        data_adjusted = data[: row - overageY, : col - overageX]
    else:
        data_adjusted = data[: row - overageY, : col - overageX, :]
    return data_adjusted
