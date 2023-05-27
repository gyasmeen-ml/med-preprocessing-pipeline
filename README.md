# med-preprocessing-pipeline

### Step 1: Download your dataset: 
Follow instructions in this link (https://github.com/neheller/kits23/tree/main/kits23)
        
        
### Step 2: Download preprocessing pipeline python scipts

``` git clone git@github.com:gyasmeen-ml/med-preprocessing-pipeline.git ``` 

``` cd med-preprocessing-pipeline ``` 

### Step 3: Data conversion and background cropping
``` python data_conversion.py --data_dir /path/to/dowbloaded/dataset/ --raw_data_dir /path/to/save/converted/data/ --crop_dir /path/to/save/background/cropped/data/ ```

### Step 4: Normalize and Resample: 
``` python resample_and_normalize.py --spacing_zxy 1. 0.78125 0.78125 --data_dir /path/to/cropped/data/from/previous/step/ --num_threads 4 --output_dir /path/to/save/resampled/and/normalized/data/ ```


### Step 5: Data loader and augmentations:     
``` python data_loader.py --patch_size_zxy 128 128 128 --data_dir /path/to/resampled/and/normalized/data/ --batch_size 4 --num_threads 4 ```
