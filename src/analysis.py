from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import tensorflow as tf
import numpy as np
import cv2

CMN_IMG_PATH = "data/COMMON_images_masks/common_{0}_image.nii.gz"

GRP_IMG_PATH = "data/g3_{0}_image.nii.gz"

LABELS = {59: {"size": 275, "pos": (63, 89)},
          60: {"size": 241, "pos": (52, 84)},
          61: {"size": 339, "pos": (97, 138)}}
N_IMG = 276 + 242 + 340

class PelvicData():
    def __init__(self, X_train_3C=None, X_train=None, y_train=None, root_path="../"):
        if X_train_3C is None and y_train is None:
            grp_indices = [59, 60, 61]
            cmn_indices = [40, 41, 42]

            self.grp_img = {}
            self.cmn_img = {}
            self.cmn_img_3c = {}


            for i in range(len(cmn_indices)):
                grp_img = sitk.ReadImage(
                    root_path + GRP_IMG_PATH.format(grp_indices[i]))
                cmn_img = sitk.ReadImage(
                    root_path + CMN_IMG_PATH.format(cmn_indices[i]))

                self.grp_img[grp_indices[i]] = sitk.GetArrayFromImage(grp_img)
                self.cmn_img[cmn_indices[i]] = sitk.GetArrayFromImage(cmn_img)
                self.cmn_img_3c[cmn_indices[i]] = np.repeat(
                    self.cmn_img[cmn_indices[i]][None, ...], 3, axis=0).T

            self.X_train = np.concatenate(list(self.grp_img.values()), axis=0)

            self.X_train_3C = np.array(
                [np.repeat(x[None, ...], 3, axis=0).T for x in self.X_train])

            self.y_train = np.zeros(N_IMG)

            i = 0
            for _, v in LABELS.items():
                self.y_train[i+v["pos"][0]:i+v["pos"][1]+1] = 1
                i += v['size']
        else:
            self.X_train_3C = X_train_3C
            self.y_train = y_train
            # self.X_train = X_train

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
