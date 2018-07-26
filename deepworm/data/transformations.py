import numpy
import skimage.transform

"""
Note: Standardization and transforms assumes
that x comes in WxHxC format from the reader
"""


def flip_horizontally(prob=0.5):
    assert 0. < prob < 1.

    def f(x,is_label=False):
        if numpy.random.random() < prob:
            return numpy.fliplr(x)
        return x

    return f


def flip_vertically(prob=0.5):
    assert 0. < prob < 1.

    def f(x,is_label=False):
        if numpy.random.random() < prob:
            
            return numpy.flipud(x)
        return x

    return f

def normalize(mean,std):
    
    def f(x,is_label=False):
        if is_label:
            x = (x-mean)/std     
        return x

    return f

def vgg_normalize():
    return normalize(numpy.array([0.485, 0.456, 0.406]),numpy.array([0.229, 0.224, 0.225]))


def rotate90(prob=0.5):
    assert 0. < prob < 1.

    def f(x,is_label=False):
        if numpy.random.random() < prob:
            return numpy.rot90(x, 2, axes=(0, 1))
        return x

    return f


def rescale(scale, **kwargs):
    """
    Rescales the image according to the scale ratio.
    :param scale: The scalar to rescale the image by.
    :param kwargs: Additional arguments for skimage.transform.resize.
    :return: The rescale function.
    """

    axes_scale = (scale, scale, 1.0)

    def f(x,is_label=False):
        return skimage.transform.\
            resize(x, numpy.multiply(x.shape, axes_scale), **kwargs)

    return f

def random_scale(scale_variance=0.2, **kwargs):
    def f(x,is_label=False):
        s = 1.+numpy.clip(scale_variance*numpy.random.randn(),-scale_variance,scale_variance)
        return skimage.transform.\
            rescale(x,s,order=0,preserve_range=True, **kwargs)
    return f

def random_contrast(contrast,clip_value=0.5):
    def f(x,is_label=False):
        if is_label:
            cont = 1.+contrast*numpy.random.randn()
            x = numpy.clip(x*cont,-clip_value,clip_value)
        return x
    return f

def random_brightness(brightness,clip_value=0.5):
    def f(x,is_label=False):
        if is_label:
            x = numpy.clip(x + brightness*numpy.random.randn(),-clip_value,clip_value)
        return x
    return f
                               
def random_transform(max_scale, max_angle, max_trans, keep_aspect_ratio=True):
    """
    Rescales the image according to the scale ratio.
    :param scale: The scalar to rescale the image by.
    :param kwargs: Additional arguments for skimage.transform.resize.
    :return: The rescale function.
    """
    
    def f(x,is_label=False):
        
        if keep_aspect_ratio:
            scalex = scaley = 1.+numpy.random.randn()*max_scale
        else:
            scalex = 1.+numpy.random.randn()*max_scale
            scaley = 1.+numpy.random.randn()*max_scale
            
        shift_y, shift_x = numpy.array(x.shape[:2]) / 2.
        shift = skimage.transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        shift_inv = skimage.transform.SimilarityTransform(translation=[shift_x+numpy.random.randn()*max_trans,
                                                                       shift_y+numpy.random.randn()*max_trans])
        trans = skimage.transform.SimilarityTransform(
            rotation=numpy.deg2rad(numpy.random.uniform(-max_angle, max_angle)),
            scale=(scalex,scaley))
        final_transform = (shift + (trans + shift_inv)).inverse
        
        return skimage.transform.warp(x, final_transform, cval=0,  preserve_range=True)

    return f

def rotate(max_angle=360):
    def f(x,is_label=False):
        # k = numpy.pi / 360 * numpy.random.uniform(-360, 360)
        k = numpy.random.uniform(-max_angle, max_angle)
        return skimage.transform.rotate(x, k)

    return f


def clip_patch(size):
    assert len(size) == 2

    def f(x,is_label=False):
        cx = numpy.random.randint(0, x.shape[0] - size[0])
        cy = numpy.random.randint(0, x.shape[1] - size[1])
        return x[cx:cx + size[0], cy:cy + size[1]]

    return f


def clip_patch_random(minsize,maxsize):
    assert len(minsize) == 2
    assert len(maxsize) == 2

    def f(x,is_label=False):
        cx = numpy.random.randint(0, x.shape[0] - f.size[0])
        cy = numpy.random.randint(0, x.shape[1] - f.size[1])
        return x[cx:cx + f.size[0], cy:cy + f.size[1]]

    def prepare():
        f.size = (numpy.random.randint(minsize[0], maxsize[0])*8, numpy.random.randint(minsize[1], maxsize[1])*8)

    f.prepare = prepare
    f.prepare()

    return f

