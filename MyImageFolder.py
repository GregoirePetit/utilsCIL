import numpy as np
import torch.utils.data as data
from PIL import Image
import  imghdr
import pandas as pd
import os, sys
import os.path
import random
import torch

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


    def __init__(self, images_list_file, transform=None, target_transform=None, return_path=False, range_classes=None, random_seed=-1, old_load=False, open_images=False, nb_classes=None):

        self.return_path = return_path
        samples = []
        df = pd.read_csv(images_list_file, sep=' ', names=['paths','class'])
        if old_load:
            root_folder = ''
        else:
            root_folder = df['paths'].head(1)[0]
            df = df.tail(df.shape[0] -1)
        df.drop_duplicates()
        df['class'] = df['class'].astype(int)
        df = df.sort_values('class')
        if nb_classes:
            order_list = [i for i in range(nb_classes)]
        else:
            order_list = [i for i in range(1+max(list(set(df['class'].values.tolist()))))]
        print('*'*(len(images_list_file)+76+len(str(random_seed))))
        print('Class order of',images_list_file,'before shuffle with seed',random_seed,': [',*order_list[:5],'...',*order_list[-5:],']')
        if random_seed != -1:
            np.random.seed(random_seed)
            random.shuffle(order_list)
            order_list = np.random.permutation(len(order_list)).tolist()
        print('Class order of',images_list_file,'after  shuffle with seed',random_seed,': [',*order_list[:5],'...',*order_list[-5:],']')
        print('*'*(len(images_list_file)+76+len(str(random_seed))))
        #print(list(set(df['class'].values.tolist())), '-> order_list')
        #print(len(list(set(df['class'].values.tolist()))), '-> len(order_list)')
        order_list_reverse = [order_list.index(i) for i in list(set(df['class'].values.tolist()))]
        if range_classes:
            index_to_take = [order_list[i] for i in range_classes]
            samples = [(os.path.join(root_folder, elt[0]),order_list_reverse[elt[1]]) for elt in list(map(tuple, df.loc[df['class'].isin(index_to_take)].values.tolist()))]
            samples.sort(key=lambda x:x[1])
            print('We pick the classes from', min(range_classes), 'to', max(range_classes), 'and we have', len(samples), 'samples')
        else:
            samples = [(os.path.join(root_folder, elt[0]),order_list_reverse[elt[1]]) for elt in list(map(tuple, df.values.tolist()))]
            samples.sort(key=lambda x:x[1])
        if not samples:
            raise(RuntimeError("No image found"))

        self.images_list_file = images_list_file
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.classes = list(set([e[1] for e in samples]))
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = [s[0] for s in samples]
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
        fmt_str += '    List Description: {}\n'.format(self.images_list_file)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class ImagesListFileFolderMixup(data.Dataset):
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


    def __init__(self, images_list_file, transform=None, target_transform=None, return_path=False, range_classes=None, random_seed=-1, old_load=False, mixup_alpha=0.2, one_every=5):

        self.return_path = return_path
        self.mixup_alpha = mixup_alpha
        self.one_every = one_every
        samples = []
        df = pd.read_csv(images_list_file, sep=' ', names=['paths','class'])
        if old_load:
            root_folder = ''
        else:
            root_folder = df['paths'].head(1)[0]
            df = df.tail(df.shape[0] -1)
        df.drop_duplicates()
        df['class'] = df['class'].astype(int)
        df = df.sort_values('class')
        order_list = [i for i in range(1+max(list(set(df['class'].values.tolist()))))]
        print('*'*(len(images_list_file)+76+len(str(random_seed))))
        print('Class order of',images_list_file,'before shuffle with seed',random_seed,': [',*order_list[:5],'...',*order_list[-5:],']')
        if random_seed != -1:
            np.random.seed(random_seed)
            random.shuffle(order_list)
            order_list = np.random.permutation(len(order_list)).tolist()
        print('Class order of',images_list_file,'after  shuffle with seed',random_seed,': [',*order_list[:5],'...',*order_list[-5:],']')
        print('*'*(len(images_list_file)+76+len(str(random_seed))))
        order_list_reverse = [order_list.index(i) for i in list(set(df['class'].values.tolist()))]
        if range_classes:
            index_to_take = [order_list[i] for i in range_classes]
            samples = [(os.path.join(root_folder, elt[0]),order_list_reverse[elt[1]]) for elt in list(map(tuple, df.loc[df['class'].isin(index_to_take)].values.tolist()))]
            samples.sort(key=lambda x:x[1])
            print('We pick the classes from', min(range_classes), 'to', max(range_classes), 'and we have', len(samples), 'samples')
        else:
            samples = [(os.path.join(root_folder, elt[0]),order_list_reverse[elt[1]]) for elt in list(map(tuple, df.values.tolist()))]
            samples.sort(key=lambda x:x[1])
        if not samples:
            raise(RuntimeError("No image found"))

        self.images_list_file = images_list_file
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.classes = list(set([e[1] for e in samples]))
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = [s[0] for s in samples]
        self.transform = transform
        self.target_transform = target_transform
        self.total_classes = 1+max(list(set(df['class'].values.tolist())))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        label = torch.zeros(self.total_classes)
        label[target] = 1.

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # perform mixup, only perform mixup roughly on 1 for every self.one_every images. the resulting can be non-mixuped because
        if index%self.one_every == 0:
            mixup_idx = random.choice([i for i in range(len(self.samples)) if i != index])
            mixup_label = torch.zeros(self.total_classes)
            label[self.targets[mixup_idx]] = 1.
            if self.transform is not None:
                mixup_image = self.transform(self.loader(self.samples[mixup_idx][0]))
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            sample = lam * sample + (1 - lam) * mixup_image
            label = lam * label + (1 - lam) * mixup_label

        if self.return_path :
            return (sample, label), self.samples[index][0]
        return sample, label

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    List Description: {}\n'.format(self.images_list_file)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class ImagesListFileFolderMultilabel(data.Dataset):
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


    def __init__(self, images_list_file, transform=None, target_transform=None, return_path=False, range_classes=None, random_seed=-1, old_load=False, open_images=False, num_labels=None):
        # images_list_file is a the root path of the list
        # transform is the transform to apply to the image
        # target_transform is the transform to apply to the target
        # return_path is a boolean to return the path of the image
        # range_classes is a range of the classes to take
        # random_seed is the seed to shuffle the classes
        # old_load is a boolean to load the list with the old way (i.e. without the root folder as first line)
        # open_images is a boolean to load the list with the open images format
        # num_labels is the number of labels to load (in case of more than 2 labels)
        self.return_path = return_path
        samples = []
        df = pd.read_csv(images_list_file, sep=' ', names=['paths','class'])
        if old_load:
            root_folder = ''
        else:
            root_folder = df['paths'].head(1)[0]
            df = df.tail(df.shape[0] -1)
        df.drop_duplicates()
        # cast the class to tuple of int
        if num_labels is None:
            df['class'] = df['class'].apply(lambda x: tuple(map(int, x.split(','))))
        else:
            df['class'] = df['class'].apply(lambda x: tuple(map(int, x.split(',')[:num_labels])))
        #print(df)
        df = df.sort_values('class')
        order_list = list(set(df['class'].values.tolist()))
        # sort the list by the first item of the tuple
        order_list.sort(key=lambda x: x[0])
        print('*'*(len(images_list_file)+76+len(str(random_seed))))
        print('Class order of',images_list_file,'before shuffle with seed',random_seed,': [',*order_list[:5],'...',*order_list[-5:],']')
        deep_copy = order_list.copy()
        if random_seed != -1:
            np.random.seed(random_seed)
            random.shuffle(order_list)
            order_list_raw = np.random.permutation(len(order_list)).tolist()
            order_list = [deep_copy[i] for i in order_list_raw]
        print('Class order of',images_list_file,'after  shuffle with seed',random_seed,': [',*order_list[:5],'...',*order_list[-5:],']')
        print('*'*(len(images_list_file)+76+len(str(random_seed))))
        order_list_reverse = [deep_copy[order_list.index(i)] for i in list(set(df['class'].values.tolist()))]
        if range_classes:
            index_to_take = [order_list[i] for i in range_classes]
            samples = [(os.path.join(root_folder, elt[0]),order_list_reverse[elt[1][0]]) for elt in list(map(tuple, df.loc[df['class'].isin(index_to_take)].values.tolist()))]
            samples.sort(key=lambda x:x[1])
            print('We pick the classes from', min(range_classes), 'to', max(range_classes), 'and we have', len(samples), 'samples')
        else:
            samples = [(os.path.join(root_folder, elt[0]),order_list_reverse[elt[1][0]]) for elt in list(map(tuple, df.values.tolist()))]
            samples.sort(key=lambda x:x[1])
            print('We have', len(samples), 'samples')
            print("first 3 samples",samples[:3])
        if not samples:
            raise(RuntimeError("No image found"))

        self.images_list_file = images_list_file
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.classes = list(set([e[1] for e in samples]))
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = [s[0] for s in samples]
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
        fmt_str += '    List Description: {}\n'.format(self.images_list_file)
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


