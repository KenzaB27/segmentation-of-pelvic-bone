import numpy as np
import SimpleITK as sitk
import registration as reg

def seg_atlas(im, atlas_ct_list, atlas_seg_list):
    """ Apply atlas-based segmentation of `im` using the list of CT images in `atlas_ct_list` and
     the corresponding segmentation masks in `atlas_seg_list`. 
     Return the resulting segmentation mask after majority voting. """
    pass


CMN_IMG_PATH = "../data/COMMON_images_masks/common_{0}_image.nii.gz"
CMN_MASK_PATH = "../data/COMMON_images_masks/common_{0}_mask.nii.gz"

GRP_IMG_PATH = "../data/g3_{0}_image.nii.gz"
GRP_MASK_PATH = "../datag3_{0}_mask.nii.gz"

class AtlasSegmentation():
    def __init__(self):
        grp_indices = [59, 60, 61]
        cmn_indices = [40, 41, 42]
        
        self.cmn_img, self.grp_img = [], []
        self.cmn_mask, self.grp_mask = [], []

        self.cmn_img_data, self.grp_img_data = [], []
        self.cmn_mask_data, self.grp_mask_data = [], []

        for i in range(len(cmn_indices)):
            
            cmn_img = sitk.ReadImage(
                CMN_IMG_PATH.format(cmn_indices[i]))
            cmn_img_data = sitk.GetArrayFromImage(cmn_img)
            self.cmn_img.append(cmn_img)
            self.cmn_img_data.append(cmn_img_data)

            cmn_mask = sitk.ReadImage(
                CMN_MASK_PATH.format(cmn_indices[i]))
            cmn_mask_data = sitk.GetArrayFromImage(cmn_mask)
            self.cmn_mask.append(cmn_mask)
            self.cmn_mask_data.append(cmn_mask_data)

            grp_img = sitk.ReadImage(
                GRP_IMG_PATH.format(grp_indices[i]))
            grp_img_data = sitk.GetArrayFromImage(grp_img)
            self.grp_img.append(grp_img)
            self.grp_img_data.append(grp_img_data)

            grp_mask = sitk.ReadImage(
                GRP_MASK_PATH.format(grp_indices[i]))
            grp_mask_data = sitk.GetArrayFromImage(grp_mask)
            self.grp_mask.append(grp_mask)
            self.grp_mask_data.append(grp_mask_data)


        #Take one image from "common" as the reference image
        ref_image = self.cmn_img[0]
        ref_mask = self.cmn_mask[0]

        #Register the CT images in "group" to that reference
        #reg.LinearTransform(im_ref_filename=COMMON_PATH + '40_image.nii.gz',im_mov_filename=COMMON_PATH + '42_image.nii.gz')

    def majority_voting(self,reg_masks):

        #Getting arrays from masks
        reg_mask_arrays = []
        for mask in reg_masks:
            reg_mask_arrays.append(sitk.GetArrayFromImage(mask))

        #Checking if 2 out of 3 agree
        majority_mask = reg_mask_arrays[0] + reg_mask_arrays[1] + reg_mask_arrays[2]
        majority_mask = np.where(majority_mask>=2, 1, 0)

        #Returning majority mask
        return  sitk.GetImageFromArray(majority_mask)






        pass