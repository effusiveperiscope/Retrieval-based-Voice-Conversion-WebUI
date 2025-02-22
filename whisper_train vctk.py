# %%
# I think that training on a single speaker makes a lot more sense than whatever
# we were trying to do earlier
CHARACTER_NAME = 'VCTKWhisperBase'
SOURCE_DIRECTORY = r'D:\DataAugmentation\VCTK-Corpus-0.92\wav48_silence_trimmed'
# Small dataset for testing
#SOURCE_DIRECTORY = r'D:\DataAugmentation\TestTwilight'
N_CPU = 16
BATCH_SIZE = 12
TOTAL_EPOCH = 1000
SAVE_EVERY_EPOCH = 10

import os 
import subprocess

model_name = CHARACTER_NAME.replace(' ', '_')

exp_dir = os.path.join('logs', model_name)
base_dir = os.path.abspath('.')
exp_dir_abs = os.path.abspath(exp_dir)
os.makedirs(exp_dir, exist_ok=True) 

SSR_DIRNAME = '3_feature1280'

# %%
# Preprocessing
print("Begin preprocessing")
subprocess.run([
    "python", os.path.join("infer", "modules", "train", "preprocess_vctk_spk.py"),
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
    "python", os.path.join("infer", "modules", "train", "extract_whisper_print.py"),
    "cuda:0", # device
    "1", # n_part (Number of GPUs),
    "0", # i_part (Starting index)
    "0", # i_gpu (Index of GPU)
    exp_dir_abs, # exp_dir
    "whisper", # version
    "False", # is_half
], capture_output=True)
print("Feature extraction done")

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
    "python", os.path.join("infer", "modules", "train", "extract_whisper_print.py"),
    "cuda:0", # device
    "1", # n_part (Number of GPUs),
    "0", # i_part (Starting index)
    "0", # i_gpu (Index of GPU)
    mute_dir, # exp_dir
    "whisper", # version
    "False", # is_half
], capture_output=True)

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

# %%

import numpy as np
print("Generating filelist...")
spk_lbls_dir = os.path.join(exp_dir_abs, "0_spk_lbls")
gt_wavs_dir = os.path.join(exp_dir_abs, "0_gt_wavs")
feature_dir = os.path.join(exp_dir_abs, SSR_DIRNAME)
f0_dir = os.path.join(exp_dir_abs, "2a_f0")
f0nsf_dir = os.path.join(exp_dir_abs, "2b-f0nsf")

sids = set()
def grab_sid(name):
    if not os.path.exists(spk_lbls_dir):
        sids.add(0)
        return 0
    with open(os.path.join(spk_lbls_dir.replace("\\","/"), name+".txt"), "r") as f:
        sid = int(f.read())
        sids.add(sid)
        return sid

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
        str(grab_sid(name))) # speaker ID which is always 0 because we have only 1 spk

# INTERESTING - they put mute items into the filelist randomly
for i in range(len(sids) + 1):
    opt.append(
        os.path.join(base_dir,"logs","mute","0_gt_wavs","mute48k.wav")+"|"+
        os.path.join(base_dir,"logs","mute",SSR_DIRNAME,"mute.npy")+"|"+
        os.path.join(base_dir,"logs","mute","2a_f0","mute.wav.npy")+"|"+
        os.path.join(base_dir,"logs","mute","2b-f0nsf","mute.wav.npy")+"|"+
        str(i)) # All speakers should be capable of generating silence
np.random.shuffle(opt)
with open(os.path.join(exp_dir_abs,"filelist.txt"),"w") as f:
    f.write("\n".join(opt))

max_sid = max(sids)
print(f"Done with {len(sids)} speakers")
print(f"Max sid: {max_sid}")

# %%
# Copy config
import shutil
shutil.copy(os.path.join("configs", "whisper", "48k.json"),
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
    # lol the descriptions for these args are reversed in the code
    #"-pg","assets/pretrained_whisper/f0G48k.pth",
    #"-pd","assets/pretrained_whisper/f0D48k.pth", 
    "-l","0",
    "-c","0", # do not cache
    "-sw","1", 
    "-v","whisper"
]))
# %%
