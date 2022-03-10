

import numpy as np
import SimpleITK as sitk
import torchio as tio
from skimage import measure, morphology
from sklearn.cluster import KMeans


def load_IMG(file_path, shape, spacing, new_spacing):
    
    dtype = np.dtype("<i2")
    fid = open(file_path, 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)

    return image


def resample(imgs, spacing, new_spacing, mode="linear"):
    """
    :return: new_image, true_spacing
    """
    dim = len(imgs.shape)
    if dim == 3 or dim == 2:
        # If the image is 3D or 2D image
        # Use torchio.Resample to resample the image.

        # Create a sitk Image object then load this object to torchio Image object
        imgs_itk = sitk.GetImageFromArray(imgs)
        imgs_itk.SetSpacing(np.flipud(spacing).astype(np.float64))
        imgs_tio = tio.ScalarImage.from_sitk(imgs_itk)
        
        # Resample Image
        resampler = tio.Resample(list(np.flipud(new_spacing)), image_interpolation=mode)
        new_imgs = resampler(imgs_tio).as_sitk()

        # Prepare return value
        new_spacing = new_imgs.GetSpacing()
        new_imgs = sitk.GetArrayFromImage(new_imgs)
        resize_factor = np.array(imgs.shape) / np.array(new_imgs.shape)
        return new_imgs, new_spacing, resize_factor
    elif dim == 4:
        # If the input is a batched 3D image
        # Run resample on each image in the batch.
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice, true_spacing, resize_factor = resample(slice, spacing, new_spacing, mode=mode)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg),[1, 2, 3, 0])
        return newimg, true_spacing, resize_factor
    else:
        raise ValueError('wrong shape')


def seg_bg_mask(img):
    """
    Calculate the segementation mask for the whole body.
    Assume the dimensions are in Superior/inferior, anterior/posterior, right/left order.
    :param img: a 3D image represented in a numpy array.
    :return: The segmentation Mask. BG = 0
    """
    (D,W,H) = img.shape

    img_cp = np.copy(img)
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(D/5):int(D/5*4),int(W/5):int(W/5*4),int(H/5):int(H/5*4)] 
    mean = np.mean(middle)  

    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # clear bg
    dilation = morphology.dilation(thresh_img,np.ones([4,4,4]))
    eroded = morphology.erosion(dilation,np.ones([4,4,4]))

    # Select the largest area besides the background
    labels = measure.label(eroded, background=1)
    regions = measure.regionprops(labels)
    roi_label = 0
    max_area = 0
    for region in regions:
        if region.label != 0 and region.area > max_area:
            max_area = region.area
            roi_label = region.label
    thresh_img = np.where(labels==roi_label, 1, 0)

    # bound the ROI. 
    # TODO: maybe should check for bounding box
    # thresh_img = 1 - eroded
    sum_over_traverse_plane = np.sum(thresh_img, axis=(1,2))
    top_idx = 0
    for i in range(D):
        if sum_over_traverse_plane[i] > 0:
            top_idx = i
            break
    bottom_idx = D-1
    for i in range(D-1, -1, -1):
        if sum_over_traverse_plane[i] > 0:
            bottom_idx = i
            break
    for i in range(top_idx, bottom_idx+1):
        thresh_img[i]  = morphology.convex_hull_image(thresh_img[i])

    labels = measure.label(thresh_img)
    
    bg_labels = []
    corners = [(0,0,0),(-1,0,0),(0,-1,0),(-1,-1,0),(0,-1,-1),(0,0,-1),(-1,0,-1),(-1,-1,-1)]
    for pos in corners:
        bg_labels.append(labels[pos])
    bg_labels = np.unique(np.array(bg_labels))
    
    mask = labels
    for l in bg_labels:
        mask = np.where(mask==l, -1, mask)
    mask = np.where(mask==-1, 0, 1)

    roi_labels = measure.label(mask, background=0)
    roi_regions = measure.regionprops(roi_labels)
    bbox = [0,0,0,D,W,H]
    for region in roi_regions:
        if region.label == 1:
            bbox = region.bbox
    
    return mask, bbox

def seg_lung_mask(img):
    """
    Calculate the segementation mask either for lung only.
    :param img: a 3D image represented in a numpy array.
    :return: The segmentation Mask.
    """
    (D,W,H) = img.shape

    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(D/5):int(D/5*4),int(W/5):int(W/5*4),int(H/5):int(H/5*4)] 
    mean = np.mean(middle)  
    img_max = np.max(img)
    img_min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==img_max]=mean
    img[img==img_min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([4,4,4]))
    dilation = morphology.dilation(eroded,np.ones([4,4,4]))

    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_regions = []
    
    for prop in regions:
        B = prop.bbox
        if (B[4]-B[1]<W/20*18 and B[4]-B[1]>W/5 and B[4]<W/20*16 and B[1]>W/10 and
                B[5]-B[2]<H/20*18 and B[5]-B[2]>H/20 and B[2]>H/10 and B[5]<H/20*19 and
                B[3]-B[0]>D/4):
            good_regions.append(prop)
            continue
            print(B)
        
        if (B[4]-B[1]<W/20*18 and B[4]-B[1]>W/6 and B[4]<W/20*18 and B[1]>W/20 and
                B[5]-B[2]<H/20*18 and B[5]-B[2]>H/20):
            good_regions.append(prop)
            continue
        
        if B[4]-B[1]<W/20*18 and B[4]-B[1]>W/20 and B[4]<W/20*18 and B[1]>W/20:
            good_regions.append(prop)
            continue
    
    # Select the most greatest region
    good_regions = sorted(good_regions, key=lambda x:x.area, reverse=True)
    
    mask = np.ndarray([D,W,H],dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    good_labels_bbox = []
    for N in good_regions[:2]:
        mask = mask + np.where(labels==N.label, 1, 0)
        good_labels_bbox.append(N.bbox)
    
    # Get the bbox of lung
    bbox = [D/2, W/2, H/2, D/2, W/2, H/2]
    for b in good_labels_bbox:
        for i in range(0, 3):
            bbox[i] = min(bbox[i], b[i])
            bbox[i+3] = max(bbox[i+3], b[i+3])
    
    mask = morphology.dilation(mask, np.ones([4,4,4])) # one last dilation
    mask = morphology.erosion(mask,np.ones([4,4,4]))

    return mask, bbox


def normalize_intensity(img, linear_clip=False, clip_range=None):
        """
        a numpy image, normalize into intensity [0,1]
        (img-img.min())/(img.max() - img.min())
        :param img: image
        :param linear_clip:  Linearly normalized image intensities so that the 95-th percentile gets mapped to 0.95; 0 stays 0
        :return:
        """

        if linear_clip:
            if clip_range is not None:
                img[img<clip_range[0]] = clip_range[0]
                img[img>clip_range[1]] = clip_range[1]
                normalized_img = (img-clip_range[0]) / (clip_range[1] - clip_range[0]) 
            else:
                img = img - img.min()
                normalized_img =img / np.percentile(img, 95) * 0.95
        else:
            # If we normalize in HU range of softtissue
            min_intensity = img.min()
            max_intensity = img.max()
            normalized_img = (img-img.min())/(max_intensity - min_intensity)
        return normalized_img
