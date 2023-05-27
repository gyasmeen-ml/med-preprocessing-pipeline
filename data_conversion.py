import sys
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
import time
from multiprocessing import Pool
from collections import OrderedDict
import SimpleITK as sitk
import numpy as np
import pickle
from copy import deepcopy

anisotropy_threshold = 3
min_feature_map_size = 4
max_numpool = 999
#YG
POSSIBLE_INPUTS_NUM = 10
ind = 0 ### which input size to select
conv_per_stage = 2
#3D
DEFAULT_BATCH_SIZE_3D = 2
unet_base_num_features_3d = 32 #Generic_UNet.BASE_NUM_FEATURES_3D
unet_max_num_filters = 320
#2D
DEFAULT_BATCH_SIZE_2D = 50
unet_base_num_features_2d = 30 # 2D
unet_max_num_filters_2d = 512

##################Training Settings
patience = 50
val_eval_criterion_alpha = 0.9  

train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
max_num_epochs = 1000
num_batches_per_epoch = 250
num_val_batches_per_epoch = 50
also_val_in_tr_mode = False
lr_threshold = 1e-6  # the network will not terminate training if the lr is still above this threshold
## Resampling and Normalization
save_every = 50
save_latest_only = True  
save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest
save_best_checkpoint = True  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
save_final_checkpoint = True  # whether or not to save the final checkpoint

        
        
        
'''
# we need to find out where the classes are and sample some random locations
# let's do 10.000 samples per class
# seed this for reproducibility!
'''

num_samples = 10000
min_percent_coverage = 0.01 # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
rndst = np.random.RandomState(1234)
RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3 
############## PART 1 #########################
def create_dataset_json(raw_data_dir, save_dir):
    
    base = raw_data_dir
    out_base = save_dir
     
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    train_patient_names = []
    test_patient_names = []
    all_cases = subfolders(base, join=False)

    train_patients = all_cases #[:489]
    test_patients = []#all_cases[:63]

    for p in train_patients:
        curr = join(base, p)
        label_file = join(curr, "segmentation.nii.gz")
        image_file = join(curr, "imaging.nii.gz")
        shutil.copy(image_file, join(imagestr, p + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelstr, p + ".nii.gz"))
        train_patient_names.append(p)
    for p in test_patients:
        curr = join(base, p)
        image_file = join(curr, "imaging.nii.gz")
        shutil.copy(image_file, join(imagests, p + "_0000.nii.gz"))
        test_patient_names.append(p)

    json_dict = {}
    json_dict['name'] = "KiTS23"
    json_dict['description'] = "kidney and kidney tumor segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "KiTS data for nnunet"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Kidney",
        "2": "Tumor",
        "3": "Cyst",
    }

    json_dict['file_ending'] = ".nii.gz"
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['data_dir'] = out_base
    json_dict['training'] = [{'image': "/imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))

############### PART 2 Crop Background ######################################
def create_lists_from_splitted_dataset(base_folder_splitted):
    lists = []

    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    for tr in training_files:
        cur_pat = []
        #cur_pat.append(join(base_folder_splitted, "imagesTr", tr['image'].split("/")[-1]))
        for mod in range(num_modalities):
            cur_pat.append(join(base_folder_splitted, "imagesTr", tr['image'].split("/")[-1][:-7] +
                                "_%04.0d.nii.gz" % mod))
        cur_pat.append(join(base_folder_splitted, "labelsTr", tr['label'].split("/")[-1]))
        lists.append(cur_pat)
    return lists, {int(i): d['modality'][str(i)] for i in d['modality'].keys()}
def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]

def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        for c in range(seg.shape[0]):
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    if seg is not None:
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    else:
        nonzero_mask = nonzero_mask.astype(int)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox

def load_crop_save(case, case_identifier, output_folder):
    try:
        ## load: read and convert into npy
        data_files = case[:-1]
        seg_file = case[-1]
        assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
        properties = OrderedDict()
        data_itk = [sitk.ReadImage(f) for f in data_files]

        properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
        properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
        properties["list_of_data_files"] = data_files
        properties["seg_file"] = seg_file

        properties["itk_origin"] = data_itk[0].GetOrigin()
        properties["itk_spacing"] = data_itk[0].GetSpacing()
        properties["itk_direction"] = data_itk[0].GetDirection()

        data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
        if seg_file is not None:
            seg_itk = sitk.ReadImage(seg_file)
            seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
        else:
            seg_npy = None
        data = data_npy.astype(np.float32)
        seg = seg_npy
        
        ## crop: crop data files to non_zero
        shape_before = data.shape
        data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
        shape_after = data.shape
        print("before crop:", shape_before, "after crop:", shape_after, "spacing:",
              np.array(properties["original_spacing"]), "\n")

        properties["crop_bbox"] = bbox
        properties['classes'] = np.unique(seg)
        seg[seg < -1] = 0
        properties["size_after_cropping"] = data[0].shape
        
        
        ## save: savr cropped data files and cropped segmentations
        all_data = np.vstack((data, seg))
        np.savez_compressed(os.path.join(output_folder, "%s.npz" % case_identifier), data=all_data)
        with open(os.path.join(output_folder, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)
    except Exception as e:
        print("Exception in", case_identifier, ":")
        print(e)
        raise edentifier


def background_crop(data_dir, cropped_dir,num_threads=16):
    
    maybe_mkdir_p(cropped_dir)
    
    lists, _ = create_lists_from_splitted_dataset(data_dir)
    get_case_identifier = lambda case : case[0].split("/")[-1].split(".nii.gz")[0][:-5]
    
    list_of_args = []
    for j, case in enumerate(lists):
        case_identifier = get_case_identifier(case)
        list_of_args.append((case, case_identifier, cropped_dir))
            
    p = Pool(num_threads)
    p.starmap(load_crop_save, list_of_args)
    p.close()
    p.join()
        
    shutil.copy(join(data_dir, "dataset.json"), cropped_dir)

###################### PART 3 Analyze Dataset ################################
def _get_voxels_in_foreground(patient_identifier, modality_id, cropped_out_dir):
        
        all_data = np.load(join(cropped_out_dir, patient_identifier) + ".npz")['data']
        data = all_data[modality_id]
        mask = all_data[-1] > 0
        voxels = list(data[mask][::10]) # no need to take every voxel
        return voxels

def _compute_stats(vox, min_perc=00.5 , max_perc=99.5):
    if len(vox) == 0:
        median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan 
    else:
        median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = np.median(vox), np.mean(vox),np.std(vox), np.min(vox), np.max(vox), np.percentile(vox, max_perc), np.percentile(vox, min_perc)       
    return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5
    
    
def analyze_dataset(cropped_out_dir, num_processes):
    #input
    patient_identifiers =  [i.split("/")[-1][:-4] for i in subfiles(cropped_out_dir, join=True, suffix=".npz")]

    #output
    ##YG REMOVEintensityproperties_file = join(cropped_out_dir, "intensityproperties.pkl") # for stats
    dataset_props_file = join(cropped_out_dir, "dataset_properties.pkl") # for stats plus other props
    ## get all spacings and sizes
    sizes = []
    spacings = []
    size_reduction = OrderedDict()
    for c in patient_identifiers:
        with open(join(cropped_out_dir, "%s.pkl" % c), 'rb') as f:
            props = pickle.load(f)
            sizes.append(props["size_after_cropping"])
            spacings.append(props["original_spacing"])

            shape_before_crop = props["original_size_of_raw_data"]
            shape_after_crop = props['size_after_cropping']
            size_red = np.prod(shape_after_crop) / np.prod(shape_before_crop)
            size_reduction[c] = size_red
    
    ## classes and modalities
    ds_json = load_json(join(cropped_out_dir, "dataset.json"))
    classes = ds_json['labels']
    all_classes = [int(i) for i in classes.keys() if int(i) > 0]
    
    modalities = ds_json["modality"]
    modalities = {int(k): modalities[k] for k in modalities.keys()}
    num_modalities = len(modalities)    
    
    
    ####YG CALCULATION TO BE USED LATER FOR NORMALIZATION
    med_size_reduction = np.median(np.array(list(size_reduction.values())))
    use_nonzero_mask_for_norm = OrderedDict()
    for i,mod in modalities.items():
        use_nonzero_mask_for_norm[i] = False if mod == 'CT' or med_size_reduction >= 3/4 else True
    for c in patient_identifiers:
        with open(join(cropped_out_dir, "%s.pkl" % c), 'rb') as f:
            props = pickle.load(f)
        props['use_nonzero_mask_for_norm'] = use_nonzero_mask_for_norm
        with open(join(cropped_out_dir, "%s.pkl" % c), 'wb') as f:
            pickle.dump(props, f)
            
    
      
    ## collect intensity information
    
    p = Pool(num_processes)
    results = OrderedDict()
    for mod_id in range(num_modalities):
        results[mod_id] = OrderedDict()
        v = p.starmap(_get_voxels_in_foreground, zip(patient_identifiers,
                                                     [mod_id] * len(patient_identifiers),
                                                    [cropped_out_dir] * len(patient_identifiers)))
        w = []
        for iv in v:
            w += iv
        np.savez(cropped_out_dir+'w.npz', w)
        median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = _compute_stats(w)
        print('*'*10,median, mean, sd, mn, mx, percentile_99_5, percentile_00_5, use_nonzero_mask_for_norm)
        
            
        local_props = p.map(_compute_stats, v)
        props_per_case = OrderedDict()
        for i, pat in enumerate(patient_identifiers):
            props_per_case[pat] = OrderedDict()
            props_per_case[pat]['median'] = local_props[i][0]
            props_per_case[pat]['mean'] = local_props[i][1]
            props_per_case[pat]['sd'] = local_props[i][2]
            props_per_case[pat]['mn'] = local_props[i][3]
            props_per_case[pat]['mx'] = local_props[i][4]
            props_per_case[pat]['percentile_99_5'] = local_props[i][5]
            props_per_case[pat]['percentile_00_5'] = local_props[i][6]
            '''For debugging
            if i == 0:
                print(pat)
                mnp = 0.5
                mxp=99.5
                for _ in range(20):
                    
                    median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = _compute_stats(w, min_perc=mnp , max_perc=mxp)
                    percentile_99_5, percentile_00_5 = 105, -45
                    print('*'*10,mnp, mxp, median, mean, sd, mn, mx, percentile_99_5, percentile_00_5, use_nonzero_mask_for_norm)
        
                    
                    data = np.load(cropped_out_dir+"/"+pat+'.npz')['data']
                    k = 100 
                    #median, mean, sd, mn, mx, percentile_99_5, percentile_00_5
                    data[0] = np.clip(data[0], percentile_00_5, percentile_99_5)
                    data[0] = (data[0] - mean) / sd

                    plt.imshow(np.squeeze(data[0, k: k + 1,...])), plt.show()
                    plt.imshow(np.squeeze(data[1, k: k + 1,...])), plt.show()
                    mnp += 0.5
                    mxp -=.5
                    break'''
                    
        
        results[mod_id]['local_props'] = props_per_case
        results[mod_id]['median'] = median
        results[mod_id]['mean'] = mean
        results[mod_id]['sd'] = sd
        results[mod_id]['mn'] = mn
        results[mod_id]['mx'] = mx
        results[mod_id]['percentile_99_5'] = percentile_99_5
        results[mod_id]['percentile_00_5'] = percentile_00_5

    p.close()
    p.join()
    
    dataset_properties = dict()
    dataset_properties['all_sizes'] = sizes
    dataset_properties['all_spacings'] = spacings
    dataset_properties['all_classes'] = all_classes
    dataset_properties['modalities'] = modalities  # {idx: modality name}
    dataset_properties['intensityproperties'] = results
    dataset_properties['size_reductions'] = size_reduction  # {patient_id: size_reduction}
    dataset_properties['use_nonzero_mask_for_norm'] = use_nonzero_mask_for_norm
    ##YG REMOVEsave_pickle(results, intensityproperties_file)
    save_pickle(dataset_properties, dataset_props_file )
    return dataset_properties


###################### PART 4 Create Config file ################################
def get_pool_conv_details(input_patch_size, current_spacing):
    ## Calculate depth of U-Net Encoder using "min_feature_map_size" and "patch_size"
    
    numpool_per_axis = np.floor([np.log(i / min_feature_map_size) / np.log(2) for i in input_patch_size]).astype(int)
    numpool_per_axis = [min(i, max_numpool) for i in numpool_per_axis]
    net_numpool = max(numpool_per_axis)
    

    ## Calculate new input shape based on network depth per axis "numpool_per_axis" and "patch_size"
    must_be_divisible_by = 2 ** np.array(numpool_per_axis)
    new_patch_size = np.zeros(len(input_patch_size))
    for i in range(len(input_patch_size)):
        if input_patch_size[i] % must_be_divisible_by[i] == 0:
            new_patch_size[i] = input_patch_size[i]
        else:
            new_patch_size[i] =  input_patch_size[i] + must_be_divisible_by[i] - (input_patch_size[i] % must_be_divisible_by[i])
        new_patch_size = new_patch_size.astype(int)

    ## Calculate num_pool conv_kernel layers based on "spacing" and "numpool_per_axis"
    dim = len(input_patch_size)
    tmp_spacing = deepcopy(current_spacing)
    reach = max(tmp_spacing)
    pool_op_kernel_sizes = []
    conv_kernel_sizes = []
    
    for p in range(net_numpool):
        reached = [tmp_spacing[i] / reach > 0.5 for i in range(dim)]
        #for debugging 
        #print(tmp_spacing, reached, [i+ p for i in numpool_per_axis])
        pool = [2 if numpool_per_axis[i] + p >= net_numpool else 1 for i in range(dim)]
        if all(reached):
            conv = [3] * dim
        else:
            conv = [3 if not reached[i] else 1 for i in range(dim)]
        pool_op_kernel_sizes.append(pool)
        conv_kernel_sizes.append(conv)
        tmp_spacing = [i * j for i, j in zip(tmp_spacing, pool)]
    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    conv_kernel_sizes.append([3] * dim)
    
    
    plan = {
            'numpool_per_axis': numpool_per_axis,
            'patch_size': new_patch_size,
            'do_dummy_2D_data_aug': (max(new_patch_size) / new_patch_size[
            0]) > anisotropy_threshold,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
            'must_be_divisible_by': must_be_divisible_by,
        }
    return plan
def get_plan_props(current_spacing, original_spacing, original_shape, data_type):
    ## Calculate input shape for UNET (try to make it isotropic then clip)
    
    new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)
    # compute how many voxels are one mm
    input_patch_size = 1 / np.array(current_spacing)
    # normalize voxels per mm
    input_patch_size /= input_patch_size.mean()
    # create an isotropic patch of size 512x512x512mm
    input_patch_size *= 1 / min(input_patch_size) * 512  # to get a starting value
    input_patch_size = np.round(input_patch_size).astype(int)
    # clip it to the median shape of the dataset because patches larger then that make not much sense
    input_patch_size = [min(i, j) for i, j in zip(input_patch_size, new_median_shape)]
    ## PRINT
    print(f'median shape (unified spacing) {new_median_shape} - isotorpic patch size after clip {input_patch_size}')

    ## Get Few Possible Input Size and Corresponding 
    possible_input_sizes = OrderedDict()
    
    if data_type != "2D":  
        batch_size = DEFAULT_BATCH_SIZE_3D
        
    else:
        batch_size = DEFAULT_BATCH_SIZE_2D
        input_patch_size, current_spacing, new_median_shape = input_patch_size[1:], current_spacing[1:], new_median_shape[1:]
        
    for i in range(POSSIBLE_INPUTS_NUM):
        possible_input_sizes[i] = get_pool_conv_details(input_patch_size, current_spacing)
        
        new_shp = possible_input_sizes[i]['patch_size']
        shape_must_be_divisible_by = possible_input_sizes[i]['must_be_divisible_by']
        
        axis_to_be_reduced = np.argsort(new_shp / new_median_shape)[-1]
        tmp = deepcopy(new_shp)
        tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
        
        input_patch_size = tmp

    
    new_patch_size = possible_input_sizes[ind]['patch_size']
    plan = {
            'batch_size': batch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'num_pool_per_axis': possible_input_sizes[ind]['numpool_per_axis'],
            'patch_size': new_patch_size,'do_dummy_2D_data_aug': (max(new_patch_size) / new_patch_size[
            0]) > anisotropy_threshold,
            'pool_op_kernel_sizes': possible_input_sizes[ind]['pool_op_kernel_sizes'],
            'conv_kernel_sizes': possible_input_sizes[ind]['conv_kernel_sizes'],
            'must_be_divisible_by': possible_input_sizes[ind]['must_be_divisible_by'],
        
            'other_possible_settings': possible_input_sizes
        }
    return plan
def create_plans(dataset_properties ,cropped_out_dir, preprocessing_output_dir, data_type = "2D"):
    list_of_npz_files = subfiles(cropped_out_dir, True, None, ".npz", True)
    
    #print("Are we using the nonzero mask for normalization?", use_nonzero_mask_for_normalization)
    spacings = dataset_properties['all_spacings']
    sizes = dataset_properties['all_sizes']
    num_cases = len(sizes)
    
    all_classes = dataset_properties['all_classes']
    modalities = dataset_properties['modalities']
    num_modalities = len(list(modalities.keys()))
    target_spacing_percentile = 70
    ## Calculate Target Spacing and account for lower resolution axis
    target_spacing = np.percentile(np.vstack(spacings), target_spacing_percentile, 0)
    
    target_size = np.percentile(np.vstack(sizes), target_spacing_percentile, 0)
    target_size_mm = np.array(target_spacing) * np.array(target_size)
    
    print(target_spacing)
    print(target_size)
    print(target_size_mm)
    
    
    # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
    # the following properties:
    # - one axis which much lower resolution than the others
    # - the lowres axis has much less voxels than the others
    # - (the size in mm of the lowres axis is also reduced)
    worst_spacing_axis = np.argmax(target_spacing)
    other_axes = [i for i in range(len(target_spacing)) if i != worst_spacing_axis]
    other_spacings = [target_spacing[i] for i in other_axes]
    other_sizes = [target_size[i] for i in other_axes]

    has_aniso_spacing = target_spacing[worst_spacing_axis] > (anisotropy_threshold * max(other_spacings))
    has_aniso_voxels = target_size[worst_spacing_axis] * anisotropy_threshold < min(other_sizes)

    
    if has_aniso_spacing and has_aniso_voxels:
        spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis] 
        target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
        # don't let the spacing of that axis get higher than the other axes
        if target_spacing_of_that_axis < max(other_spacings):
            target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
        target_spacing[worst_spacing_axis] = target_spacing_of_that_axis 
    
    ## Calculate new shapes of each data volumes after unifying spacing 
    new_shapes = [(np.array(old_spacing) / target_spacing) * np.array(old_shape) for old_spacing, old_shape in zip(spacings, sizes)]

    ## PRINT
    median_shape = np.median(np.vstack(new_shapes), 0)
    print("the median shape of the dataset is ", median_shape)
    
    max_shape = np.max(np.vstack(new_shapes), 0)
    print("the max shape in the dataset is ", max_shape)
    min_shape = np.min(np.vstack(new_shapes), 0)
    print("the min shape in the dataset is ", min_shape)

    print("we don't want feature maps smaller than ", min_feature_map_size, " in the bottleneck")

    
    ## Calculate forward and backward transpose (depth to be access 0 --> lower resolution, i.e. highest spacing)
    max_spacing_axis = np.argmax(target_spacing)
    remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
    transpose_forward = [max_spacing_axis] + remaining_axes
    transpose_backward = [np.argwhere(np.array(transpose_forward) == i)[0][0] for i in range(3)]
    ## PRINT
    median_spacing_t = np.array(target_spacing)[transpose_forward]
    median_shape_t = np.array(median_shape)[transpose_forward]
    print("the transposed median shape of the dataset is ", median_shape_t)
    
    ## Plan Stage 1
    original_spacing = median_spacing_t
    current_spacing = median_spacing_t
    original_shape = median_shape_t
    
    plans_per_stage = list()
    plans_per_stage.append( get_plan_props(current_spacing, original_spacing, original_shape, data_type))
    print(plans_per_stage[0]['median_patient_size_in_voxels'])
    if data_type == "2D":
        unet_base_num_features = unet_base_num_features_2d
    else:
        unet_base_num_features = unet_base_num_features_3d
        
        ## Plan Stage 0
        lowres_stage_spacing = deepcopy(target_spacing)
        num_voxels = np.prod(median_shape, dtype=np.float64)

        current_median_patient_size = plans_per_stage[0]['median_patient_size_in_voxels']
        while 2 * np.prod(current_median_patient_size, dtype=np.int64) >= np.prod(
                    plans_per_stage[0]['median_patient_size_in_voxels'], dtype=np.int64):
            max_spacing = max(lowres_stage_spacing)
            
            if np.any((max_spacing / lowres_stage_spacing) > 2):
                #print('****' , max_spacing , lowres_stage_spacing, (max_spacing / lowres_stage_spacing)>2, lowres_stage_spacing)
                lowres_stage_spacing[(max_spacing / lowres_stage_spacing) > 2] \
                            *= 1.01
            else:
                lowres_stage_spacing *= 1.01
            num_voxels = np.prod(target_spacing / lowres_stage_spacing * median_shape, dtype=np.float64)
            lowres_stage_spacing_t = np.array(lowres_stage_spacing)[transpose_forward]

            current_median_patient_size = np.round(median_spacing_t / lowres_stage_spacing_t * median_shape_t).astype(int)

            #print(current_median_patient_size, lowres_stage_spacing)


        plans_per_stage.append(get_plan_props(lowres_stage_spacing_t, median_spacing_t,median_shape_t, data_type))

    plans_per_stage = plans_per_stage[::-1]
    plans_per_stage = {i: plans_per_stage[i] for i in range(len(plans_per_stage))}  # convert to dict
   

    ## Normalization Scheme:
    schemes = OrderedDict()
    for i in range(num_modalities):
        if modalities[i] == "CT" or modalities[i] == 'ct':
            schemes[i] = "CT"
        elif modalities[i] == 'noNorm':
            schemes[i] = "noNorm"
        else:
            schemes[i] = "nonCT"
                
                
    plans = {'num_stages': len(list(plans_per_stage.keys())), 
             'num_modalities': num_modalities,
             'modalities': modalities, 
             'normalization_schemes': schemes,
             'dataset_properties': dataset_properties, 
             'list_of_npz_files': list_of_npz_files,
             'original_spacings': spacings, 
             'original_sizes': sizes,
             'preprocessed_data_folder': preprocessing_output_dir, 
             'num_classes': len(all_classes),
             'all_classes': all_classes, 
             'base_num_features': unet_base_num_features,
             'use_mask_for_norm': dataset_properties['use_nonzero_mask_for_norm'],
             'keep_only_largest_region': None,
             'min_region_size_per_class': None, 
             'min_size_per_class': None,
             'transpose_forward': transpose_forward, 
             'transpose_backward': transpose_backward,
             'plans_per_stage': plans_per_stage,
             'conv_per_stage': conv_per_stage,
            }
    # Save Plans
    with open(preprocessing_output_dir +f'/config.pkl', 'wb') as f:
        pickle.dump(plans, f)
    
    return plans

if __name__ == "__main__":
    '''
    python data_conversion.py --data_dir /mnt/hdd/sda/ygeo/code/kits23/kits23_project/dataset/ --raw_data_dir /mnt/hdd/sda/ygeo/tmp_dataset_processed/ --crop_dir /mnt/hdd/sda/ygeo/tmp_dataset_processed/cropped/
    '''
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, help = "path to downloaded nii dataset", default =  "/mnt/hdd/sda/ygeo/code/kits23/kits23_project/dataset/")
    parser.add_argument("--raw_data_dir",  type=str, default ="/mnt/hdd/sda/ygeo/tmp_dataset_processed/")

    parser.add_argument("--crop_dir",  type=str, default = "/mnt/hdd/sda/ygeo/tmp_dataset_processed/cropped/")
    parser.add_argument("--num_threads",  type=int, default = 8)
    args = parser.parse_args()
    
    ########### PART 1
    create_dataset_json(args.data_dir, args.raw_data_dir )

    ########### PART 2
    strt = time.time()
    background_crop( args.raw_data_dir,  args.crop_dir, num_threads=args.num_threads)
    print(f'Cropping zero voxels finished in {time.time() - strt} seconds')
    
    ########### PART 3
    strt = time.time()
    dataset_properties = analyze_dataset(args.crop_dir,args.num_threads)
    print(f'Save Dataset Information finished in {time.time() - strt} seconds')
    
    strt = time.time()
    dataset_properties = load_pickle(join(args.crop_dir, "dataset_properties.pkl"))
    plans = create_plans(dataset_properties, args.crop_dir, args.crop_dir, data_type="3D")
    print(f'Create Training Configs finished in {time.time() - strt} seconds')
