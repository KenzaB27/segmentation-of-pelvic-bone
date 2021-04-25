import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy import optimize
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from scipy.interpolate import griddata


def read_image(file_name):
    img = sitk.ReadImage(file_name)
    img = sitk.Cast(img, sitk.sitkFloat32)
    img_data = sitk.GetArrayFromImage(img)
    return img, img_data


def plot_3d_img(img, img_data, cmap="gray"):
    imageSize = img.GetSize()
    plt.figure(figsize=(12, 8))
    plt.subplot(131)
    plt.imshow(img_data[int(imageSize[0]/2), :, :], cmap=cmap)
    plt.subplot(132)
    plt.title("3D image of a pelvic", size=16)
    plt.imshow(img_data[:, int(imageSize[1]/2), :], cmap=cmap)
    plt.subplot(133)
    plt.imshow(img_data[:, :, int(imageSize[2]/2)], cmap=cmap)
    plt.show()


def ssd(img1, img2):
    dist = np.sum((img1 - img2) ** 2)
    dist /= float(img1.shape[0] * img1.shape[1])
    return dist


def mi(img1, img2, nbins=100):
    hist, _ = np.histogramdd(
        [np.ravel(img1), np.ravel(img2)], bins=nbins)
    hist /= np.sum(hist)
    H1 = entropy(np.sum(hist, axis=0))
    H2 = entropy(np.sum(hist, axis=1))
    H12 = entropy(np.ravel(hist))
    dist = - (H1 + H2) / H12
    return dist

class Transform():
    def __init__(self, im_ref, im_mov, errtype='SSD'):
        super().__init__()
        self.im_ref = im_ref
        self.im_mov = im_mov
        self.errtype = errtype

    def apply_transf(self, par):
        pass

    def error_transf(self, par):
        transf_img = self.apply_transf(par)
        if self.errtype == 'SSD':
            error = ssd(self.im_ref, transf_img)
        elif self.errtype == 'MI':
            error = mi(self.im_ref, transf_img)
        else:
            raise Exception("Errtype not defined")
        return error

class LinearTransform(Transform):
    def __init__(self, im_ref, im_mov, errtype='SSD'):
        super().__init__(im_ref, im_mov, errtype)

    def apply_transf(self, lin_xfm):
        """ Apply given linear transform `lin_xfm` to `im_mov` and 
        return the transformed image. """
        scale = np.array([[lin_xfm[0], 0], [0, lin_xfm[1]]])
        rot = np.array([[np.cos(lin_xfm[2]), np.sin(lin_xfm[2])],
                        [-np.sin(lin_xfm[2]), np.cos(lin_xfm[2])]])
        shear = np.array([[1, 0], [lin_xfm[3], 1]])
        transf_matrix = scale @ rot @ shear
        transf_img = ndi.interpolation.affine_transform(
            self.im_mov, transf_matrix, offset=[lin_xfm[4], lin_xfm[5]])
        return transf_img

    def est_transf(self, method='Powell'):
        """ Estimate linear transform to align `im_mov` to `im_ref` and 
        return the transform parameters. """
        initial_guess = np.array([1, 1, 0, 0, 0, 0], np.float32)
        result = optimize.minimize(
            self.error_transf, initial_guess, method=method, options={'disp': True})
        return result.x


class NonLinearTransform(Transform):
    def __init__(self, im_ref, im_mov, errtype):
        super(NonLinearTransform, self).__init__(im_ref, im_mov, errtype)

    def apply_transf(self, nl_xfm):
        """ Apply given non-linear transform `nl_xfm` to `im_mov` and 
        return the transformed image."""
        pass

    def est_transf(self, method='Powell'):
        """ Estimate non-linear transform to align `im_mov` to `im_ref` and 
        return the transform parameters. """
        pass

def seg_atlas(im, atlas_ct_list, atlas_seg_list):
    """ Apply atlas-based segmentation of `im` using the list of CT images in `atlas_ct_list` and
     the corresponding segmentation masks in `atlas_seg_list`. 
     Return the resulting segmentation mask after majority voting. """
    pass


def train_classifier(im_list, labels_list):
    """ Receive a list of images `im_list` and a list of vectors (one per image) with the labels 0 or 1 depending on the sagittal 
    2D slice contains or not the pubic symphysis. 
    Returns the trained classifier.
    """
    pass


def pubic_symphysis_selection(im, classifier):
    """ Receive a CT image and the trained classifier. 
    Returns the sagittal slice number with the maximum probability of containing the pubic symphysis."""
    pass
