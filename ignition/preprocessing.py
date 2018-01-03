from .imports import *


def isotropically_resize_img(img, size=256, side="shortest", resample=PIL.Image.BICUBIC):
    """Resizes an image while keeping the aspect ratio intact.

    Parameters
    ----------
    img: a PIL Image instance
    size: the new size in pixels of the specified side
    side: either "shortest" or "longest"
    resample: the resampling filter to use: PIL.Image.NEAREST, 
        PIL.Image.BILINEAR, PIL.Image.BICUBIC, or PIL.Image.LANCZOS

    Returns
    -------
    A PIL Image instance.

    Raises
    ------
    ValueError: if invalid `side` is passed.
    """
    w, h = img.size
    if side == "shortest":
        other = max(w, h) * size // min(w, h)
        shape = (other, size) if w > h else (size, other)
    elif side == "longest":
        other = min(w, h) * size // max(w, h)
        shape = (size, other) if w > h else (other, size)
    else:
        raise ValueError("Unknown side: %s" % side)
    return img.resize(shape, resample)


def crop_img(img, left, top, width, height):
    """Crops the image to the specified rectangle.
    
    Parameters
    ----------
    img: a PIL Image instance
    left, top, width, height: the crop rectangle. This rectangle
        may lie outside the image.

    Returns
    -------
    A PIL Image instance.
    """
    return img.crop((left, top, left + width, top + height))


def randomly_crop_img(img, width, height, seed=None):
    """Crops a random rectangle from the image.

    Parameters
    ----------
    img: a PIL Image instance
    width, height: the size of the crop rectangle
    seed: random seed

    Returns
    -------
    A PIL Image instance.

    Raises
    ------
    ValueError: if the crop rectangle is larger than the image.
    """
    x_range = img.size[0] - width + 1
    y_range = img.size[1] - height + 1
    if x_range < 1 or y_range < 1:
        raise ValueError("Crop rectangle (%d, %d) cannot be larger than image %s" % (width, height, img.size))

    if seed is not None:
        np.random.seed(seed)

    left = np.random.randint(x_range)
    top = np.random.randint(y_range)
    return crop_img(img, left, top, width, height)


def color_jitter(img, seed=None):
    """Randomly manipulates the contrast, brightness, and color balance.

    Parameters
    ----------
    img: a PIL Image instance
    seed: random seed

    Returns
    -------
    A PIL Image instance.
    """
    if seed is not None:
        np.random.seed(seed)

    def contrast(img):
        return PIL.ImageEnhance.Contrast(img).enhance(np.random.random() + 0.5)

    def brightness(img):
        return PIL.ImageEnhance.Brightness(img).enhance(np.random.random() + 0.5)

    def color_balance(img):
        return PIL.ImageEnhance.Color(img).enhance(np.random.random() + 0.5)

    funcs = [contrast, brightness, color_balance]
    funcs = np.random.choice(funcs, size=len(funcs), replace=False)
    for fn in funcs:
        img = fn(img)
    return img

