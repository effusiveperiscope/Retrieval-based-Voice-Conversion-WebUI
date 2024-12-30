# %%
CHARACTER_NAME = 'FACodecBase'
SOURCE_DIRECTORY = r'D:\DataAugmentation\TITAN-Medium-Dataset'
N_CPU = 16
BATCH_SIZE = 16
TOTAL_EPOCH = 1000
SAVE_EVERY_EPOCH = 5

import os 
import subprocess

model_name = CHARACTER_NAME.replace(' ', '_')

exp_dir = os.path.join('logs', model_name)
base_dir = os.path.abspath('.')
exp_dir_abs = os.path.abspath(exp_dir)
os.makedirs(exp_dir, exist_ok=True) 

SSR_DIRNAME = '3_feature2'

# %%
# Preprocessing
print("Begin preprocessing")
subprocess.run([
    "python", os.path.join("infer", "modules", "train", "preprocess.py"),
    SOURCE_DIRECTORY, # inp_root
    str(48000), # sr
    str(N_CPU), # n_p
    exp_dir_abs, # exp_dir
    "False", # noparallel
    str(3.7) # per (?)
], env = os.environ) 
print("Preprocessing done")

# %%
# F0 Extraction
print("Begin F0 extraction")
subprocess.run([
    "python", os.path.join("infer", "modules", "train", "extract", "extract_f0_rmvpe.py"),
    "1", # n_part (Number of GPUs),
    "0", # i_part (Starting index)
    "0", # i_gpu (Index of GPU)
    exp_dir_abs, # exp_dir
    "False" # is_half
    ])
print("F0 extraction done")

# %%
# Feature extraction
print("Begin feature extraction")
subprocess.run([
    "python", os.path.join("infer", "modules", "train", "extract_facodec_print.py"),
    "cuda:0", # device
    "1", # n_part (Number of GPUs),
    "0", # i_part (Starting index)
    "0", # i_gpu (Index of GPU)
    exp_dir_abs, # exp_dir
    "facodec", # version
    "False", # is_half
], capture_output=True)
print("Feature extraction done")

# %%
# Z-normalize features for numerical stability
print("Z-normalizing features...")
import numpy as np
from tqdm import tqdm
for feat in tqdm(os.listdir(os.path.join(exp_dir_abs, SSR_DIRNAME))):
    f = np.load(os.path.join(exp_dir_abs, SSR_DIRNAME, feat))
    np.save(os.path.join(exp_dir_abs, SSR_DIRNAME, feat), (f-np.mean(f))/np.std(f))
print("Done")

# %%
# Calculate features for mute
mute_dir = os.path.abspath(os.path.join('logs','mute'))
subprocess.run([
    "python", os.path.join("infer", "modules", "train", "extract", "extract_f0_rmvpe.py"),
    "1", # n_part (Number of GPUs),
    "0", # i_part (Starting index)
    "0", # i_gpu (Index of GPU)
    mute_dir, # exp_dir
    "False" # is_half
    ])
subprocess.run([
    "python", os.path.join("infer", "modules", "train", "extract_facodec_print.py"),
    "cuda:0", # device
    "1", # n_part (Number of GPUs),
    "0", # i_part (Starting index)
    "0", # i_gpu (Index of GPU)
    mute_dir, # exp_dir
    "facodec", # version
    "False", # is_half
], capture_output=True)
for feat in tqdm(os.listdir(os.path.join(mute_dir, SSR_DIRNAME))):
    f = np.load(os.path.join(mute_dir, SSR_DIRNAME, feat))
    np.save(os.path.join(mute_dir, SSR_DIRNAME, feat), (f-np.mean(f))/np.std(f))

# %%
# Inspect shapes
import numpy as np
f0 = os.listdir(os.path.join(exp_dir_abs, '2a_f0'))
ssr = os.listdir(os.path.join(exp_dir_abs, SSR_DIRNAME))
f0_example = np.load(os.path.join(exp_dir_abs, '2a_f0', f0[0]))
ssr_example = np.load(os.path.join(exp_dir_abs, SSR_DIRNAME, ssr[0]))
print(f0[0])
print(f0_example.shape)
print(ssr[0])
print(ssr_example.shape)
print(f"ssr mean: {np.mean(ssr_example)}")

# %%
import numpy as np
print("Generating filelist...")
gt_wavs_dir = os.path.join(exp_dir_abs, "0_gt_wavs")
feature_dir = os.path.join(exp_dir_abs, SSR_DIRNAME)
f0_dir = os.path.join(exp_dir_abs, "2a_f0")
f0nsf_dir = os.path.join(exp_dir_abs, "2b-f0nsf")
names = (
    set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
    & set([name.split(".")[0] for name in os.listdir(feature_dir)])
    & set([name.split(".")[0] for name in os.listdir(f0_dir)])
    & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
) # Kind of scary set intersection

opt = []
for name in names:
    opt.append(
        os.path.join(gt_wavs_dir.replace("\\","/"),name)+".wav|"+
        os.path.join(feature_dir.replace("\\","/"),name)+".npy|"+
        os.path.join(f0_dir.replace("\\","/"),name)+".wav.npy|"+
        os.path.join(f0nsf_dir.replace("\\","/"),name)+".wav.npy|"+
        "0") # speaker ID which is always 0 because we have only 1 spk

# INTERESTING - they put mute items into the filelist randomly
for _ in range(2):
    opt.append(
        os.path.join(base_dir,"logs","mute","0_gt_wavs","mute48k.wav")+"|"+
        os.path.join(base_dir,"logs","mute",SSR_DIRNAME,"mute.npy")+"|"+
        os.path.join(base_dir,"logs","mute","2a_f0","mute.wav.npy")+"|"+
        os.path.join(base_dir,"logs","mute","2b-f0nsf","mute.wav.npy")+"|"+
        "0")
np.random.shuffle(opt)
with open(os.path.join(exp_dir_abs,"filelist.txt"),"w") as f:
    f.write("\n".join(opt))
print("Done")
# %%
# Copy config
import shutil
shutil.copy(os.path.join("configs", "facodec", "48k.json"),
    os.path.join(exp_dir_abs, "config.json"))

# %%
# Train
print(' '.join([
    "python",
    os.path.join("infer","modules","train","train.py"),
    "-e",model_name,
    "-sr","48k",
    "-f0","1",
    "-bs",str(BATCH_SIZE),
    "-g","0", # GPU index to use
    "-te",str(TOTAL_EPOCH),
    "-se",str(SAVE_EVERY_EPOCH),
    # NO pretrained model, we are training a new base
    "-l","1",
    "-c","0", # 11hr data, do not cache
    "-sw","1", # Save weights
    "-v","facodec"
]))
# Avoid automatically doing this, will need to do it manually while we debug
# subprocess.run([
    # "python",
    # os.path.join("infer","modules","train","train.py"),
    # "-e",model_name,
    # "-sr","48k",
    # "-f0","1",
    # "-bs",str(BATCH_SIZE),
    # "-g","1", # Number of GPUs
    # "-te",str(TOTAL_EPOCH),
    # "-se",str(SAVE_EVERY_EPOCH),
    ## NO pretrained model, we are training a new base
    # "-l","1",
    # "-c","0", # 11hr data, do not cache
    # "-sw","0",
    # "-v","facodec"
# ])
# %%
