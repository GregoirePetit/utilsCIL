import os
import sys
import pandas as pd

if len(sys.argv) != 2:
    print('Arguments: images_dir')
    sys.exit(-1)

images_dir = sys.argv[1]
toto = []
for subfolder in os.listdir(images_dir):
    if os.path.isdir(os.path.join(images_dir, subfolder)):
        for image in os.listdir(os.path.join(images_dir, subfolder)):
            if image.endswith('.jpg'):
                toto.append((os.path.join(images_dir, subfolder, image), int(subfolder[1:])))
# sort toto by class
toto = sorted(toto, key=lambda x: x[1])
for i in toto:
    print(i[0], i[1])