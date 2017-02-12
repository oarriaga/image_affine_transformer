import glob
import numpy as np
from scipy.misc import imread, imresize

class ImageTransfomer(object):
    """Affine transformation class with bilinear interpolation for batches
    of images with shape (batch_size, height, width, num_channels)."""

    def __init__(self, images_path, shape):
        self.image_path = images_path
        self.images = self._load_images_from_path(self.image_path,
                                                            shape)
        self.batch_size = self.images.shape[0]
        self.height = self.images.shape[1]
        self.width = self.images.shape[2]
        self.num_channels = self.images.shape[3]
        self.transformed_images = None

    def _list_files(self, directory_name):
        return glob.glob(directory_name)

    def _load_image(self, image_filename, shape=None):
        image_array = imread(image_filename)
        if shape != None:
            image_array = imresize(image_array, shape)
        #image_array = np.expand_dims(image_array,0)
        return image_array

    def _load_images_from_path(self, images_path, shape):
        image_filenames = self._list_files(images_path)
        images = []
        for image_name in image_filenames:
            image = self._load_image(image_name, shape)
            images.append(image)
        images = np.asarray(images)
        return images

    def _create_grids_of_indices(self):
        x_linspace = np.linspace(-1, 1, self.width)
        y_linspace = np.linspace(-1, 1, self.height)
        x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
        ones = np.ones(np.prod(x_coordinates.shape))
        indices_grid = np.vstack([x_coordinates.flatten(),
                                   y_coordinates.flatten(),
                                   ones])
        indices_grid = np.expand_dims(indices_grid, axis=0)
        indices_grids = np.repeat(indices_grid, self.batch_size, axis=0)
        return indices_grids

    def _create_transformations(self, transformation):
        transformation = np.expand_dims(transformation,0)
        transformations = np.repeat(transformation, self.batch_size, axis=0)
        return transformations

    def _apply_affine_transformations(self, transformations, indices_grids):
        transformed_indices_grids = np.matmul(transformations, indices_grids)
        transformed_indices_grids = transformed_indices_grids.reshape(
                                                    self.batch_size, 2 ,
                                                    self.height, self.width)
        transformed_indices_grids = np.moveaxis(transformed_indices_grids,
                                                                    1, -1)
        return transformed_indices_grids

    def _transform_indices_to_image_size(self, transformed_indices_grids):
        x_transformed_indices = transformed_indices_grids[:, :, :, 0]
        y_transformed_indices = transformed_indices_grids[:, :, :, 1]
        x_transformed_indices = .5 *(x_transformed_indices + 1.0) * self.width
        y_transformed_indices = .5 *(y_transformed_indices + 1.0) * self.height
        return x_transformed_indices, y_transformed_indices

    def _get_indices_of_corners(self, x_transformed_indices,
                                   y_transformed_indices):

        #00: lower left; 01: upper left; 10: lower right; 11: upper right
        corners_indices_00 = np.floor(x_transformed_indices).astype(np.int64)
        corners_indices_01 = np.floor(y_transformed_indices).astype(np.int64)
        corners_indices_10 = corners_indices_00 + 1
        corners_indices_11 = corners_indices_01 + 1

        corners_indices_00 = np.clip(corners_indices_00, 0, self.width - 1)
        corners_indices_01 = np.clip(corners_indices_01, 0, self.height - 1)
        corners_indices_10 = np.clip(corners_indices_10, 0, self.width - 1)
        corners_indices_11 = np.clip(corners_indices_11, 0, self.height - 1)

        indices_of_corners = (corners_indices_00, corners_indices_10,
                                corners_indices_01, corners_indices_11)

        return indices_of_corners

    def _get_pixel_values_of_corners(self, indices_of_corners):
        x0, x1, y0, y1 = indices_of_corners
        batch_matrix = np.arange(self.batch_size)[:, None, None]
        pixel_values_00 = self.images[batch_matrix, y0, x0]
        pixel_values_01 = self.images[batch_matrix, y1, x0]
        pixel_values_10 = self.images[batch_matrix, y0, x1]
        pixel_values_11 = self.images[batch_matrix, y1, x1]
        pixel_values_of_corners = (pixel_values_00, pixel_values_01,
                                    pixel_values_10, pixel_values_11)
        return pixel_values_of_corners

    def _calculate_areas(self,x_transformed_indices,
                        y_transformed_indices,
                        indices_of_corners):

        x0, x1, y0, y1 = indices_of_corners
        areas_a = (x1 - x_transformed_indices) * (y1 - y_transformed_indices)
        areas_b = (x1 - x_transformed_indices) * (y_transformed_indices - y0)
        areas_c = (x_transformed_indices - x0) * (y1 - y_transformed_indices)
        areas_d = (x_transformed_indices - x0) * (y_transformed_indices - y0)

        areas_a = np.expand_dims(areas_a, axis=3)
        areas_b = np.expand_dims(areas_b, axis=3)
        areas_c = np.expand_dims(areas_c, axis=3)
        areas_d = np.expand_dims(areas_d, axis=3)

        areas = (areas_a, areas_b, areas_c, areas_d)
        return areas

    def _transform_images(self, pixel_values_of_corners, areas):
        corners_00, corners_01, corners_10, corners_11 = (
                                                    pixel_values_of_corners)
        areas_a, areas_b, areas_c, areas_d = areas
        transformed_images = (areas_a*corners_00 + areas_b*corners_01 +
                        areas_c*corners_10 + areas_d*corners_11)
        transformed_images = transformed_images.astype('uint8')
        return transformed_images

    def interpolate_all_images(self, affine_transformation):
        indices_grids = self._create_grids_of_indices()
        transformations = self._create_transformations(affine_transformation)
        transformed_indices_grids = self._apply_affine_transformations(
                                                            transformations,
                                                            indices_grids)
        transformed_indices = self._transform_indices_to_image_size(
                                                    transformed_indices_grids)
        x_transformed_indices, y_transformed_indices = transformed_indices
        indices_of_corners = self._get_indices_of_corners(
                                                        x_transformed_indices,
                                                        y_transformed_indices)
        pixel_values_of_corners = self._get_pixel_values_of_corners(
                                                        indices_of_corners)
        areas = self._calculate_areas(x_transformed_indices,
                                        y_transformed_indices,
                                        indices_of_corners)
        self.transformed_images = self._transform_images(
                                                pixel_values_of_corners, areas)
