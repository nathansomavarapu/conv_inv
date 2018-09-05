import os
from PIL import Image
import glob
import tqdm
import warnings

# warnings.filterwarnings('error')

imgs = glob.glob('data/*/*/*')

for img in tqdm.tqdm(imgs):
    try:
        img_t = Image.open(img)
        img_t.verify()
        if img_t is None:
            os.remove(img)
        img_t.close()
    except:
        os.remove(img)
