def est_lin_transf(im_ref, im_mov):
    """ Estimate linear transform to align `im_mov` to `im_ref` and 
    return the transform parameters. """
    pass


def est_nl_transf(im_ref, im_mov):
    """ Estimate non-linear transform to align `im_mov` to `im_ref` and 
    return the transform parameters. """
    pass


def apply_lin_transf(im_mov, lin_xfm):
    """ Apply given linear transform `lin_xfm` to `im_mov` and 
    return the transformed image. """
    pass


def apply_nl_transf(im_mov, nl_xfm):
    """ Apply given non-linear transform `nl_xfm` to `im_mov` and 
    return the transformed image."""
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
