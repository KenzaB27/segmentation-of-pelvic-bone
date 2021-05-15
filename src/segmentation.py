import numpy as np
import SimpleITK as sitk
from registration import LinearTransform, NonLinearTransform, Transform

CMN_IMG_PATH = "../data/COMMON_images_masks/common_{0}_image.nii.gz"
CMN_MASK_PATH = "../data/COMMON_images_masks/common_{0}_mask.nii.gz"

GRP_IMG_PATH = "../data/g3_{0}_image.nii.gz"
GRP_MASK_PATH = "../data/g3_{0}_mask.nii.gz"

class AtlasSegmentation():
    def __init__(self):
        grp_indices = [59, 60, 61]
        cmn_indices = [40, 41, 42]
        
        self.cmn_img, self.grp_img = {}, {}
        self.cmn_mask, self.grp_mask = {}, {}

        self.cmn_img_data, self.grp_img_data = {}, {}
        self.cmn_mask_data, self.grp_mask_data = {}, {}

        for i in range(len(cmn_indices)):
            
            self.cmn_img[cmn_indices[i]] = sitk.ReadImage(
                CMN_IMG_PATH.format(cmn_indices[i]))
            self.cmn_img_data[cmn_indices[i]] = sitk.GetArrayFromImage(
                self.cmn_img[cmn_indices[i]])

            self.cmn_mask[cmn_indices[i]] = sitk.ReadImage(
                CMN_MASK_PATH.format(cmn_indices[i]))
            self.cmn_mask_data[cmn_indices[i]] = sitk.GetArrayFromImage(
                self.cmn_mask[cmn_indices[i]])

            self.grp_img[grp_indices[i]] = sitk.ReadImage(
                GRP_IMG_PATH.format(grp_indices[i]))
            self.grp_img_data[grp_indices[i]] = sitk.GetArrayFromImage(
                self.grp_img[grp_indices[i]])

            self.grp_mask[grp_indices[i]] = sitk.ReadImage(
                GRP_MASK_PATH.format(grp_indices[i]))
            self.grp_mask_data[grp_indices[i]] = sitk.GetArrayFromImage(
                self.grp_mask[grp_indices[i]])

    def majority_voting(self, reg_masks):
        """Provide the majority voting result for a list of segmented images. The images must be nd-arrays"""
        pass

    def seg_atlas(self, id_cmn):
        """ Apply atlas-based segmentation of `im` using the list of CT images in `atlas_ct_list` and
        the corresponding segmentation masks in `atlas_seg_list`. 
        Return the resulting segmentation mask after majority voting. """
        reg_mask = []
        for id_grp in self.grp_img:
            lin_trf = LinearTransform(
                im_ref=self.cmn_img[id_cmn], im_mov=self.grp_img[id_grp])
            lin_xfm = lin_trf.est_transf(metric="MI", num_iter=200)
            
            lin_reg_img = lin_trf.apply_transf(lin_xfm)
            lin_reg_mask = lin_trf.apply_transf(transformation=lin_xfm, im=self.grp_mask[id_grp])

            nl_trf = NonLinearTransform(
                im_ref=self.cmn_img[id_cmn], im_mov=lin_reg_img)
            nl_xfm = nl_trf.est_transf(metric="SSD", num_iter=200)

            nl_reg_mask = nl_trf.apply_transf(
                transformation=nl_xfm, im=lin_reg_mask)

            reg_mask.append(nl_reg_mask)

        return self.majority_voting(reg_mask)
