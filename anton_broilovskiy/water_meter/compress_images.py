"""File to compless images to blosc format"""
import os
import sys
import dill

import blosc
from tqdm import tqdm
import matplotlib.pyplot as plt




def compress(argv=None):
    """Convert images from any format to blosc format
    Can use with aruments from command line.
    Fist argument - path_from. It's a path with images.
    Second argumnet - path_to. It's a path with created blosc files.
    flag -d delete all images
    If path_from and path_to is empty - images and blosc files will create in working directory."""
    if argv is None:
        argv = sys.argv

    try:
        dell = True if '-d' in argv else False
        path_from, path_to = [argv[1], argv[2]] if len(argv) - int(dell) >= 3 else [argv[1], argv[1]]

    except IndexError:
        path_from = './'
        path_to = './'

    print('path from: %s'%path_from)
    print('path to: %s\n'%path_to)
    files_name = os.listdir(path_from)
    for imfile in tqdm(files_name):
        impath = os.path.join(path_from, imfile)
        if not os.path.isdir(impath):
            image = plt.imread(impath)
            image_path = os.path.join(path_to, imfile[:-4])
            with open(image_path + '.blosc', 'w+b') as file_blosc:
                file_blosc.write(blosc.compress(dill.dumps(image)))
            if dell:
                os.remove(impath)

if __name__ == "__main__":
    sys.exit(compress())
