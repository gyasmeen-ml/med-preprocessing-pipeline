import json
import glob
import pickle
import time
import numpy as np
from tqdm import tqdm
classmap = {}


if __name__ == "__main__":                    
    
    #run example: python create_json_classmap --which_dataset kits --pkl_dir "path/to/dir/with/pickle/files" --json_save_path "path/to/save/json/file/"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_dir",  type=str, default = "/mnt/hdd/sda/ygeo/data/Untitled Folder/kits21/kits21/data/preprocessed/transformed/plan2D_stage0/", help="directory for pkl files")
    parser.add_argument("--json_save_path", type = str, default = "/mnt/hdd/sda/ygeo/data/Untitled Folder/kits21/kits21/data/preprocessed/transformed/plan2D_stage0/", help="output directory")
    parser.add_argument("--which_dataset", choices=['kits', 'chaos', 'sabs'], type = str, default = "kits", help="output directory")
    args = parser.parse_args()
    save_dir = args.json_save_path 
    data_root = args.pkl_dir

    MIN_TP=1 # minimum number of true positive pixels in a slice
    
    #data_root = "/mnt/hdd/sda/ygeo/data/CHAOST2/Train_Sets/chaos_MR_T2_normalized_512/"
    #LABEL_NAME = ["BG", "LIVER", "RK", "LK", "SPLEEN"]     


    #data_root = "/mnt/hdd/sda/ygeo/data/MultiAtlas_CT_Labelling/Abdomen/Abdomen/RawData/Training/sabs_CT_normalized_512/"
    #LABEL_NAME = ["BGD", "SPLEEN", "KID_R", "KID_l", "GALLBLADDER", "ESOPHAGUS", "LIVER", "STOMACH", "AORTA", "IVC", "PS_VEIN", "PANCREAS", "AG_R", "AG_L"]     

    #data_root = "/mnt/hdd/sda/ygeo/data/Untitled Folder/kits21/kits21/data/preprocessed/transformed/plan2D_stage0/"
    #LABEL_NAME = ["background","kidney","tumor", "cyst"]

    
    if args.which_dataset == "kits":
        LABEL_NAME = ["background","kidney","tumor", "cyst"]
    elif args.which_dataset == "chaos":
        LABEL_NAME = ["BG", "LIVER", "RK", "LK", "SPLEEN"]  
    elif args.which_dataset == "sabs":
        LABEL_NAME = ["BGD", "SPLEEN", "KID_R", "KID_l", "GALLBLADDER", "ESOPHAGUS", "LIVER", "STOMACH", "AORTA", "IVC", "PS_VEIN", "PANCREAS", "AG_R", "AG_L"]    
    else:
        print('wrong dataset')
    
    
    file_list = glob.glob(data_root + "/image_*.pkl")

    fid = f'{save_dir}/classmap_{MIN_TP}_mod.json'
    for _lb in LABEL_NAME:
        classmap[_lb] = {}
        for FNAME in (file_list):
            pid, slc = FNAME[:-4].split('_')[-1].split('-z')
            classmap[_lb][str(pid)] = []

    c = 0

    p_arr = np.zeros(len(file_list))
    for FNAME in tqdm(file_list):
        t = time.time()
        pid, slc = FNAME[:-4].split('_')[-1].split('-z')
        p_arr[c] = int(pid) 
        c += 1

        with open(FNAME, "rb") as f:
            _ = pickle.load(f)
            lbl_real = pickle.load(f)

        for cls in range(len(LABEL_NAME)):
            if cls in np.unique(lbl_real):
                if np.sum(lbl_real == cls) >= MIN_TP:
                    classmap[LABEL_NAME[cls]][str(pid)].append(int(slc))
        t = time.time() - t
        print(f"Processing {c} out of {len(file_list)} in {t}: file name: {pid}-z{slc}")

        #print(f'pid {str(pid)} slice {slc} finished!')

    print(len(np.unique(p_arr)))

    with open(fid, 'w') as fopen:
        json.dump(classmap, fopen)
        fopen.close()  

