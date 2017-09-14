from . import image

from keras import backend as K


def euclidean(vectors):
    x, y = vectors
    return K.sqrt(K.sum(K.square(x - y), axis=(1, 2)))


def gram_matrix(x, norm_by_channels=False):
    """
    Returns the Gram matrix of the tensor x.
    """
    if K.ndim(x) == 2:
        # Flatten batches are up-sampled again.
        x = K.expand_dims(K.expand_dims(x, 1), 1)

    if K.ndim(x) == 3:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        shape = K.shape(x)
        C, H, W = shape[0], shape[1], shape[2]
        gram = K.dot(features, K.transpose(features))
    elif K.ndim(x) == 4:
        # Swap from (H, W, C) to (B, C, H, W)
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]
        # Reshape as a batch of 2D matrices with vectorized channels
        features = K.reshape(x, K.stack([B, C, H * W]))
        # This is a batch of Gram matrices (B, C, C).
        gram = K.batch_dot(features, features, axes=2)
    else:
        raise ValueError('The input tensor should be either a 3d (H, W, C) or 4d (B, H, W, C) tensor.')
    # Normalize the Gram matrix
    if norm_by_channels:
        denominator = C * H * W  # Normalization from Johnson
    else:
        denominator = H * W  # Normalization from Google
    gram = gram / K.cast(denominator, x.dtype)

    return gram


def get_style_features(out_dict, layer_names, norm_by_channels=False):
    features = []
    for l in layer_names:
        x = out_dict[l]
        features.append((x))
    return features
