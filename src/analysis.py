from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import SimpleITK as sitk
import tensorflow as tf
import numpy as np

CMN_IMG_PATH = "data/COMMON_images_masks/common_{0}_image.nii.gz"

GRP_IMG_PATH = "data/g3_{0}_image.nii.gz"

LABELS = {59: {"size": 275, "pos": (63, 89)},
          60: {"size": 241, "pos": (52, 84)},
          61: {"size": 339, "pos": (97, 138)}}
N_IMG = 276 + 242 + 340


class PelvicData():
    def __init__(self, root_path="../", split=0.05):

        grp_indices = [59, 60, 61]
        cmn_indices = [40, 41, 42]

        # Create Labels
        i = 0
        self.y_train = np.zeros(N_IMG)
        for _, v in LABELS.items():
            self.y_train[i+v["pos"][0]:i+v["pos"][1]+1] = 1
            i += v['size']

        # Create Features
        self.grp_img = {}
        self.cmn_img = {}
        self.cmn_img_3c = {}

        for i in range(len(cmn_indices)):
            grp_img = sitk.ReadImage(
                root_path + GRP_IMG_PATH.format(grp_indices[i]))
            cmn_img = sitk.ReadImage(
                root_path + CMN_IMG_PATH.format(cmn_indices[i]))

            self.grp_img[grp_indices[i]] = sitk.GetArrayFromImage(grp_img)[
                :, 100:400, 100:400]
            self.cmn_img[cmn_indices[i]] = sitk.GetArrayFromImage(cmn_img)[
                :, 100:400, 100:400]
            self.cmn_img_3c[cmn_indices[i]] = np.array(
                [np.repeat(x[None, ...], 3, axis=0).T for x in self.cmn_img[cmn_indices[i]]])

        self.X_train = np.concatenate(list(self.grp_img.values()), axis=0)

        # Preprocess data (scale 0-1)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=split, random_state=42, stratify=self.y_train)

        self.X_flat_train = np.array([x.flatten() for x in self.X_train])
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_flat_train)
        self.X_flat_train = self.scaler.transform(self.X_flat_train)

        self.X_flat_val = np.array([x.flatten() for x in self.X_val])
        self.X_flat_val = self.scaler.transform(self.X_flat_val)

        self.X_train = self.X_flat_train.reshape(self.X_train.shape)
        self.X_val = self.X_flat_val.reshape(self.X_val.shape)

        self.X_train_3C = np.array(
            [np.repeat(x[None, ...], 3, axis=0).T for x in self.X_train])
        self.X_val_3C = np.array(
            [np.repeat(x[None, ...], 3, axis=0).T for x in self.X_val])




