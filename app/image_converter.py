"""Convert tabular network-flow features into image representations."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64


def normalize_features(features):
    """Normalize feature values to the range [0, 1]."""
    features = np.asarray(features, dtype=float)
    min_value = np.min(features)
    max_value = np.max(features)

    if max_value == min_value:
        return np.zeros_like(features, dtype=float)

    return (features - min_value) / (max_value - min_value)


def feature_to_rgb(value):
    """Map a normalized scalar value to a 24-bit RGB tuple."""
    rgb_value = int(float(value) * 16777215)
    return (rgb_value >> 16) & 0xFF, (rgb_value >> 8) & 0xFF, rgb_value & 0xFF


def features_to_rgb_image(features, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH):
    """Convert numeric features into a 64x64 RGB image array."""
    normalized_features = normalize_features(features)
    total_pixels = image_height * image_width
    pixels_per_feature = max(1, total_pixels // len(normalized_features))
    output_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    pixel_index = 0
    for feature_value in normalized_features:
        rgb = feature_to_rgb(feature_value)
        for _ in range(pixels_per_feature):
            if pixel_index >= total_pixels:
                break
            x = pixel_index // image_width
            y = pixel_index % image_width
            output_image[x, y] = rgb
            pixel_index += 1

    while pixel_index < total_pixels:
        x = pixel_index // image_width
        y = pixel_index % image_width
        output_image[x, y] = feature_to_rgb(normalized_features[-1])
        pixel_index += 1

    return output_image


def features_to_grayscale_image(features, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH):
    """Convert numeric features into an interpolated 64x64 grayscale image array."""
    normalized_features = normalize_features(features)
    features_per_dim = int(np.ceil(np.sqrt(len(normalized_features))))

    x = np.linspace(0, features_per_dim - 1, features_per_dim)
    y = np.linspace(0, features_per_dim - 1, features_per_dim)
    grid = np.zeros((features_per_dim, features_per_dim))

    for index, value in enumerate(normalized_features):
        row = index // features_per_dim
        col = index % features_per_dim
        grid[row, col] = value

    interpolator = RegularGridInterpolator(
        (x, y), grid, method="linear", bounds_error=False, fill_value=None
    )

    x_new = np.linspace(0, features_per_dim - 1, image_width)
    y_new = np.linspace(0, features_per_dim - 1, image_height)
    x_mesh, y_mesh = np.meshgrid(x_new, y_new, indexing="ij")

    return interpolator((x_mesh.ravel(), y_mesh.ravel())).reshape(image_height, image_width)
