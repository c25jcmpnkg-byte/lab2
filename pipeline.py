import os
import glob
from ultralytics import YOLO
from PIL import Image

frames = sorted(glob.glob("frames/frame_*.jpg"))

det_model = YOLO("yolov8n.pt")
seg_model = YOLO("yolov8n-seg.pt")

os.makedirs("output_detection", exist_ok=True)
os.makedirs("output_segmentation", exist_ok=True)

for f in frames:
    r = det_model(f, verbose=False)
    Image.fromarray(r[0].plot()[..., ::-1]).save("output_detection/" + os.path.basename(f))

    r = seg_model(f, verbose=False)
    Image.fromarray(r[0].plot()[..., ::-1]).save("output_segmentation/" + os.path.basename(f))

for folder, name in [("output_detection", "detection.gif"), ("output_segmentation", "segmentation.gif")]:
    files = sorted(glob.glob(folder + "/frame_*.jpg"))
    imgs = [Image.open(f) for f in files]
    imgs[0].save(name, save_all=True, append_images=imgs[1:], duration=200, loop=0)