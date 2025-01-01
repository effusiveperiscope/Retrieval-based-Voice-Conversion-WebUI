# %%
import torch
import soundfile as sf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_half = False
test_wav = 'logs/FACodecBase/1_16k_wavs/3180_1.wav'

wav, sr = sf.read(test_wav)

# %% 
# Setup FACodec
from infer.modules.vc.modules import load_facodec

config = {'is_half': is_half, 'device': device}
facodec_model = load_facodec(config)

# %% 
# Extract features
from einops import rearrange
twav = torch.from_numpy(wav).to(device)
if twav.dim() == 2:  # stereo channels
    twav = twav.mean(-1)
if is_half:
    twav = twav.half()
else:
    twav = twav.float()
print(twav.shape)
twav = twav.unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    feats = facodec_model.encoder(twav)
    vq_post_emb, vq_id, _, quantized, spk_embs = facodec_model.decoder(
        feats)
    content_code = vq_id[1:3]
    feats = rearrange(content_code, 'c b t -> b t c')
    feats = feats.to(torch.float32)
    print(feats.mean(), feats.std())
    feats = (
        feats - feats.mean()) / feats.std()
    print(feats.shape)
    print(feats.mean(), feats.std())

# %%
print(feats)

# %%
