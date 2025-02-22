import multiprocessing
import os
import sys

from scipy import signal

now_dir = os.getcwd()
sys.path.append(now_dir)
print(sys.argv)
inp_root = sys.argv[1]
sr = int(sys.argv[2])
n_p = int(sys.argv[3])
exp_dir = sys.argv[4]
noparallel = sys.argv[5] == "True"
per = float(sys.argv[6])
import multiprocessing
import os
import re
import traceback

inp_root = os.path.abspath(inp_root)

import librosa
import numpy as np
from scipy.io import wavfile

from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer

mutex = multiprocessing.Lock()
f = open("%s/preprocess.log" % exp_dir, "a+")


def println(strr):
    mutex.acquire()
    print(strr)
    f.write("%s\n" % strr)
    f.flush()
    mutex.release()

def walkdir(r):
    for root, dirs, files in os.walk(r):
        for file in files:
            if file.endswith(".wav") or file.endswith(".flac"):
                yield os.path.join(root, file) # Will yield absolute path

class PreProcess:
    def __init__(self, sr, exp_dir, per=3.7):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.spk_lbls_dir = "%s/0_spk_lbls" % exp_dir
        self.gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
        self.wavs16k_dir = "%s/1_16k_wavs" % exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.spk_lbls_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1, sid):
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5: # This filters out audio that is too loud (?)
            print("%s-%s-%s-filtered" % (idx0, idx1, tmp_max))
            return
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio # This looks like a volume normalization
        wavfile.write(
            "%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000
        )  # , res_type="soxr_vhq"
        wavfile.write(
            "%s/%s_%s.wav" % (self.wavs16k_dir, idx0, idx1),
            16000,
            tmp_audio.astype(np.float32),
        )
        with open("%s/%s_%s.txt" % (self.spk_lbls_dir, idx0, idx1), "w") as f:
            f.write("%s" % sid) # Kind of wasteful but it works

    def pipeline(self, path, idx0, sid):
        try:
            audio = load_audio(path, self.sr)
            # zero phased digital filter cause pre-ringing noise...
            # audio = signal.filtfilt(self.bh, self.ah, audio)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1, sid)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        idx1 += 1
                        break
                self.norm_write(tmp_audio, idx0, idx1, sid)
            println("%s->Suc." % path)
        except:
            println("%s->%s" % (path, traceback.format_exc()))

    def pipeline_mp(self, infos):
        for path, idx0, sid in infos:
            self.pipeline(path, idx0, sid)

    def speaker_map(self, fpath):
        # VCTK: Map speakers
        # s5 = 0
        # p225 = 1
        # p376 = 152
        vctk_prefix = 'wav48_silence_trimmed'
        fpath = fpath.replace('\\', '/') # normalize path for windows
        if (vctk_prefix + '/' + 's5') in fpath:
            return 0
        psearch = re.search(r'wav48_silence_trimmed/p(\d{3})/', fpath)
        if psearch:
            return int(psearch.group(1)) - 224
        import pdb
        pdb.set_trace()
        return None

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            infos = [
                (fpath, idx, self.speaker_map(fpath))
                for idx, fpath in enumerate(sorted(list(walkdir(inp_root))))
            ]

            if noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p],)
                    )
                    ps.append(p)
                    p.start()
                for i in range(n_p):
                    ps[i].join()
        except:
            println("Fail. %s" % traceback.format_exc())


def preprocess_trainset(inp_root, sr, n_p, exp_dir, per):
    pp = PreProcess(sr, exp_dir, per)
    println("start preprocess")
    println(sys.argv)
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    println("end preprocess")


if __name__ == "__main__":
    preprocess_trainset(inp_root, sr, n_p, exp_dir, per)
