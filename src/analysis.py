from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import tensorflow as tf
import numpy as np
import cv2

CMN_IMG_PATH = "../data/COMMON_images_masks/common_{0}_image.nii.gz"

GRP_IMG_PATH = "../data/g3_{0}_image.nii.gz"

LABELS = {59: {"size": 275, "pos": (63, 89)},
          60: {"size": 241, "pos": (52, 84)},
          61: {"size": 339, "pos": (97, 138)}}
N_IMG = 276 + 242 + 340


def whitening(image):
    """Whitening. Normalises image to zero mean and unit variance."""

    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret

class PelvicData():
    def __init__(self, train_X_3C=None, train_X=None, train_y=None):
        if train_X_3C is None and train_y is None:
            grp_indices = [59, 60, 61]
            cmn_indices = [40, 41, 42]

            self.grp_img = {}
            self.cmn_img = {}

            for i in range(len(cmn_indices)):
                grp_img = sitk.ReadImage(
                    GRP_IMG_PATH.format(grp_indices[i]))
                cmn_img = sitk.ReadImage(
                    CMN_IMG_PATH.format(cmn_indices[i]))

                self.grp_img[grp_indices[i]] = sitk.GetArrayFromImage(grp_img)
                self.cmn_img[cmn_indices[i]] = sitk.GetArrayFromImage(cmn_img)

            self.train_X = np.concatenate(list(self.grp_img.values()), axis=0)

            self.train_X_3C = np.array(
                [np.repeat(x[None, ...], 3, axis=0).T for x in self.train_X])

            self.train_y = np.zeros(N_IMG)

            i = 0
            for _, v in LABELS.items():
                self.train_y[i+v["pos"][0]:i+v["pos"][1]+1] = 1
                i += v['size']
        else:
            self.train_X_3C = train_X_3C
            self.train_y = train_y
            self.train_X = train_X
        

        


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


def dice_analysis(mask, ref_mask):
    """Checking and fixing Origin and Spacing, Maybe this is not how this should be fixed. """
    if mask.GetSpacing() != ref_mask.GetSpacing():
        ref_mask.SetSpacing(mask.GetSpacing())

    if mask.GetOrigin() != ref_mask.GetOrigin():
        ref_mask.SetOrigin(mask.GetOrigin())

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(ref_mask, mask)
    result = overlap_measures_filter.GetDiceCoefficient()

    print(result)
    return result


def hausdorf_distance_analysis(mask, ref_mask):
    """Checking and fixing Origin and Spacing, Maybe this is not how this should be fixed. """
    if mask.GetSpacing() != ref_mask.GetSpacing():
        ref_mask.SetSpacing(mask.GetSpacing())

    if mask.GetOrigin() != ref_mask.GetOrigin():
        ref_mask.SetOrigin(mask.GetOrigin())

    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    hausdorff_distance_filter.Execute(ref_mask, mask)
    result = hausdorff_distance_filter.GetHausdorffDistance()
    print(result)
    return result
