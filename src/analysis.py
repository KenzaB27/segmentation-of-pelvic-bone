import SimpleITK as sitk

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
