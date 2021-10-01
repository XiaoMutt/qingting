import os

from PIL import Image
from tqdm import tqdm

from core.config import COLLECTED_BACKGROUND_IMAGE_SIZE
from core.etl.utils import gray_and_crop
from scripts.config import COLLECTED_BACKGROUNDS_PATH, FLICKR30K_IMAGE_PATH

if __name__ == '__main__':
    with open(COLLECTED_BACKGROUNDS_PATH, 'wb') as writer:
        for file in tqdm(os.listdir(FLICKR30K_IMAGE_PATH)):
            if file.endswith('.jpg'):
                with Image.open(os.path.join(FLICKR30K_IMAGE_PATH, file)) as img:
                    img = gray_and_crop(img, COLLECTED_BACKGROUND_IMAGE_SIZE)
                    if img:
                        b = img.tobytes()
                        writer.write(b)
