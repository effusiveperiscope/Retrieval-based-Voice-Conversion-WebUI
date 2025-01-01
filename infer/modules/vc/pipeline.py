import os
import sys
import traceback
import logging
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

from functools import lru_cache
from time import time as ttime

import faiss
import librosa
import soundfile as sf
import numpy as np
import parselmouth
import pyworld
import torch
import torch.nn.functional as F
import torchcrepe
from scipy import signal
from einops import rearrange

now_dir = os.getcwd()
sys.path.append(now_dir)

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}


@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    # print(data1.max(),data2.max())
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class Pipeline(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000  # facodec SR
        self.window = 100  # hop length for facodec
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值
        self.device = config.device

    def get_f0(
        self,
        input_audio_path,
        x,
        p_len,
        f0_up_key,
        f0_method,
        filter_radius,
        inp_f0=None,
        target_f0_mean=None
    ):
        global input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            if input_audio_path is None:
                raise ValueError('Null input_audio_path not supported with harvest')
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, 10)
            if filter_radius > 2:
                f0 = signal.medfilt(f0, 3)
        elif f0_method == "crepe":
            model = "full"
            # Pick a batch size that doesn't cause memory errors on your gpu
            batch_size = 512
            # Compute pitch using first gpu
            audio = torch.tensor(np.copy(x))[None].float()
            f0, pd = torchcrepe.predict(
                audio,
                self.sr,
                self.window,
                f0_min,
                f0_max,
                model,
                batch_size=batch_size,
                device=self.device,
                return_periodicity=True,
            )
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[pd < 0.1] = 0
            f0 = f0[0].cpu().numpy()
        elif f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                from infer.lib.rmvpe import RMVPE
                logger.info("Loading rmvpe model")
                rvc_rmvpe_path = '%s/rmvpe.pt' % os.environ.get('rmvpe_root', '.')
                self.model_rmvpe = RMVPE(
                    model_path=rvc_rmvpe_path,
                    is_half=self.is_half,
                    device=self.device,
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

            if "privateuseone" in str(self.device):  # clean ortruntime memory
                del self.model_rmvpe.model
                del self.model_rmvpe
                logger.info("Cleaning ortruntime memory")
        elif f0_method == "forward":
            f0 = x

        f0 *= pow(2, f0_up_key / 12)
        if target_f0_mean is not None:
            cur_mean = (f0[f0 != 0]).mean()
            f0 *= target_f0_mean / cur_mean
        # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        tf0 = self.sr // self.window  # 每秒f0点数
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]
        # with open("test_opt.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak  # 1-0

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
        extra_hooks
    ):  # ,file_index,file_big_npy
        feats = torch.from_numpy(audio0)
        t0 = ttime()
        if extra_hooks.get('feature_override') is None:
            feats = feats.float() # Need 32 bit precision for FACodec always
            # 16 bit is too temperamental
            if feats.dim() == 2:  # double channels
                feats = feats.mean(-1)
            assert feats.dim() == 1, feats.dim()
            feats = feats.unsqueeze(0).unsqueeze(0)

            facodec_model = model
            with torch.no_grad():
                feats = feats.to(self.device)
                feats = facodec_model.encoder(feats)
                vq_post_emb, vq_id, _, quantized, spk_embs = facodec_model.decoder(
                    feats)
                content_code = vq_id[1:3]
                feats = rearrange(content_code, 'c b t -> b t c')

                feats = feats.to(torch.float32)
                feats = (feats - feats.mean()) / feats.std()
        else:
            feature_override = extra_hooks.get('feature_override')
            feats = feature_override(feats).to(self.device) # Pass in the padded audio

        # from PyQt5.QtCore import pyqtRemoveInputHook
        # import pdb
        # pyqtRemoveInputHook()
        # pdb.set_trace()

        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()

        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if (
            not isinstance(index, type(None))
            and not isinstance(big_npy, type(None))
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            # _, I = index.search(npy, 1)
            # npy = big_npy[I.squeeze()]

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)

        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            arg = (feats, p_len, pitch, pitchf, sid) if hasp else (feats, p_len, sid)

            audio1 = (net_g.infer(*arg)[0][0, 0]).data.cpu().float().numpy()

            del hasp, arg
        del feats, p_len
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def vc_tts(
        self,
        net_g,
        sid,
        infer_dict,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):  # ,file_index,file_big_npy
        t0 = ttime()
        with torch.no_grad():
            feats = infer_dict["features"][0].to(self.device)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if (
            not isinstance(index, type(None))
            and not isinstance(big_npy, type(None))
            and index_rate != 0
        ):
            npy = feats.cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            # _, I = index.search(npy, 1)
            # npy = big_npy[I.squeeze()]

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        p_len = infer_dict["f0"].size(1)
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        
        p_len = torch.tensor([p_len], device=self.device).long()
        if self.is_half:
            feats = feats.type(torch.float16)
        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            arg = (feats, p_len, pitch, pitchf, sid) if hasp else (feats, p_len, sid)
            audio1 = (net_g.infer(*arg)[0][0, 0]).data.cpu().float().numpy()
            del hasp, arg
        del feats, p_len
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        input_audio_path,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        f0_file=None,
        extra_hooks={},
        target_f0_mean=None
    ):
        if (
            file_index != ""
            # and file_big_npy != ""
            # and os.path.exists(file_big_npy) == True
            and os.path.exists(file_index)
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                # big_npy = np.load(file_big_npy)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i : i - self.window])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        audio_sum[t - self.t_query : t + self.t_query]
                        == audio_sum[t - self.t_query : t + self.t_query].min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        #sf.write('test_audio_pad.wav', audio_pad, samplerate=16000)

        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except:
                traceback.print_exc()
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
                input_audio_path,
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                filter_radius,
                inp_f0,
                target_f0_mean=target_f0_mean
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if "mps" not in str(self.device) or "xpu" not in str(self.device):
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        t2 = ttime()
        times[1] += t2 - t1
        for t in opt_ts:
            t = t // self.window * self.window
            if if_f0 == 1:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                        extra_hooks
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                        extra_hooks
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if if_f0 == 1:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                    extra_hooks
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                    extra_hooks
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if tgt_sr != resample_sr >= 16000:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt

    def pipeline_tts(
        self,
        net_g,
        sid,
        infer_dict,
        f0_up_key,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        version,
        protect,
        f0_file=None,
        extra_hooks={},
    ):
        if (
            file_index != ""
            and os.path.exists(file_index)
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None
        audio_opt = []
        inp_f0 = None
        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except:
                traceback.print_exc()
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            p_len = infer_dict["f0"].size(1)
            pitch, pitchf = self.get_f0(
                "",
                infer_dict["f0"].numpy(),
                p_len,
                f0_up_key,
                "forward",
                filter_radius,
                inp_f0,
                target_f0_mean=target_pitch
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if "mps" not in str(self.device) or "xpu" not in str(self.device):
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).long()
            pitchf = torch.tensor(pitchf, device=self.device).float()
        audio_opt.append(
                self.vc_tts(
                    net_g,
                    sid,
                    infer_dict,
                    pitch,
                    pitchf,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )
        )
        audio_opt = np.concatenate(audio_opt)
        if tgt_sr != resample_sr >= 16000:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt
