import os
import torch
from PIL import Image
from tqdm import tqdm

from AltClip import AltCLIP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from glob import glob

download_path = os.path.join("../AltClip/checkpoints", "altclip-xlmr-l")
model = getattr(AltCLIP, "AltCLIP").from_pretrain(
    download_path="../AltClip/checkpoints",
    model_name="altclip-xlmr-l",
    only_download_config=False,
    device="cpu", )
process = getattr(AltCLIP, "AltCLIPProcess").from_pretrained(download_path)
transform = process.feature_extractor
tokenizer = process.tokenizer

model.eval()
model.to(device)
videos = glob("video_frames/*/")
save_path = "video_features/altclip_features"

for video in tqdm(videos, colour="green"):
    video_features = []
    raw_images: list = glob(f"{video}/*.jpg")
    raw_images.sort(key=lambda x: int(x.split("\\")[-1].replace(".jpg", "")))
    for image in raw_images:
        img = Image.open(image)
        img = transform(img)
        img = torch.tensor(img["pixel_values"]).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(img).cpu()
        video_features.append(image_features)
    video_features = torch.cat(video_features)
    video_name = video.split("\\")[-2]
    torch.save(video_features, f"{save_path}/{video_name}.pt")
