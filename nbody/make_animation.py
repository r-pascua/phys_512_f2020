import glob
import os
import sys
from PIL import Image

directory = sys.argv[1]
try:
    time_per_frame = float(sys.argv[2])
except (IndexError, ValueError):
    time_per_frame = 5
files = glob.glob(os.path.join(directory, "*.png"))
files = sorted(files, key=lambda f: int(f[f.rindex("_")+1:-4]))
frames = []
for fn in files:
    frames.append(Image.open(fn))

frames[0].save(
    os.path.join(directory, "animation.gif"),
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=time_per_frame * len(files),
    loop=0
)
