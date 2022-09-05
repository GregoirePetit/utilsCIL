import numpy as np
import torch.utils.data as data
from PIL import Image
import  imghdr
import pandas as pd
import os, sys
import os.path
import random


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)



class ImagesListFileFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """


    def __init__(self, images_list_file, transform=None, target_transform=None, return_path=False, range_classes=None, random_seed=-1, old_load=False):

        self.return_path = return_path
        samples = []
        df = pd.read_csv(images_list_file, sep=' ', names=['paths','class'])
        if old_load:
            root_folder = ''
        else:
            root_folder = df['paths'].head(1)[0]
            df = df.tail(df.shape[0] -1)
        df.drop_duplicates()
        #df = df.sample(frac=0.15)
        df = df.sort_values('class')
        order_list = [i for i in range(1+max(list(set(df['class'].values.tolist()))))]
        print(order_list[:5],order_list[-5:])
        if random_seed != -1:
            np.random.seed(random_seed)
            #random.seed(random_seed)
            random.shuffle(order_list)
            order_list = np.random.permutation(len(order_list)).tolist()
        #order_list=[53, 37, 65, 51, 4, 20, 38, 9, 10, 81, 44, 36, 84, 50, 96, 90, 66, 16, 80, 33, 24, 52, 91, 99, 64, 5, 58, 76, 39, 79, 23, 94, 30, 73, 25, 47, 31, 45, 19, 87, 42, 68, 95, 21, 7, 67, 46, 82, 11, 6, 41, 86, 88, 70, 18, 78, 71, 59, 43, 61, 22, 14, 35, 93, 56, 28, 98, 54, 27, 89, 1, 69, 74, 2, 85, 40, 13, 75, 29, 34, 92, 0, 77, 55, 49, 3, 62, 12, 26, 48, 83, 60, 57, 63, 15, 32, 8, 97, 72, 17]
        #order_list=[68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]
        #order_list=[34, 37, 71, 58, 2, 70, 28, 17, 75, 25, 82, 77, 55, 8, 6, 91, 87, 64, 52, 40, 11, 36, 10, 90, 38, 88, 47, 74, 94, 20, 26, 53, 81, 54, 78, 48, 72, 66, 5, 12, 83, 30, 16, 43, 93, 97, 84, 76, 98, 31, 92, 50, 69, 67, 27, 3, 86, 7, 60, 23, 59, 46, 62, 1, 68, 63, 99, 22, 49, 15, 32, 96, 80, 41, 95, 13, 18, 9, 29, 65, 24, 0, 56, 39, 85, 35, 89, 45, 73, 51, 19, 44, 42, 21, 14, 4, 57, 33, 79, 61]
        print(order_list[:5],order_list[-5:])
        order_list_reverse = [order_list.index(i) for i in list(set(df['class'].values.tolist()))]
        if range_classes:
            index_to_take = [order_list[i] for i in range_classes]
            samples = [(os.path.join(root_folder, elt[0]),order_list_reverse[elt[1]]) for elt in list(map(tuple, df.loc[df['class'].isin(index_to_take)].values.tolist()))]
            samples.sort(key=lambda x:x[1])
        else:
            samples = [(os.path.join(root_folder, elt[0]),order_list_reverse[elt[1]]) for elt in list(map(tuple, df.values.tolist()))]
            samples.sort(key=lambda x:x[1])
        if not samples:
            raise(RuntimeError("No image found"))

        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.classes = list(set([e[1] for e in samples]))
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_path :
            return (sample, target), self.samples[index][0]
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str




IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


#index image file folder

class IndexImagesListFileFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, images_list_file, transform=None, target_transform=None, return_path=False):

        self.return_path = return_path
        images_list_file = open(images_list_file, 'r').readlines()
        samples = []
        for e in images_list_file:
            e = e.strip()
            image_path = e.split()[0]
            try:
                assert (os.path.exists(image_path))
            except AssertionError:
                print('Cant find ' + image_path)
                sys.exit(-1)
            image_class = int(e.split()[-1])
            samples.append((image_path, image_class))

        if len(samples) == 0:
            raise (RuntimeError("No image found"))

        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.classes = list(set([e[1] for e in samples]))
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_path:
            return index, self.samples[index][0], sample, target
        return index, sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
