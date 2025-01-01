import os

#from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


from ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download
import torch

# FACodec model
class FACodecModel:
    def __init__(self, config):
        if hasattr(config, "device"):
            self.device = config.device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = FACodecEncoder(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )
        self.decoder = FACodecDecoder(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )

        encoder_ckpt = hf_hub_download(
            repo_id="amphion/naturalspeech3_facodec",
             filename="ns3_facodec_encoder.bin")
        decoder_ckpt = hf_hub_download(
            repo_id="amphion/naturalspeech3_facodec",
             filename="ns3_facodec_decoder.bin")

        self.encoder.load_state_dict(torch.load(encoder_ckpt))
        self.decoder.load_state_dict(torch.load(decoder_ckpt))

        self.encoder.eval()
        self.decoder.eval()
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        if config.is_half:
            self.encoder = self.encoder.half()
            self.decoder = self.decoder.half()
        else:
            self.encoder = self.encoder.float()
            self.decoder = self.decoder.float()

def load_facodec(config):
    return FACodecModel(config)