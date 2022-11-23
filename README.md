# utilsCIL
Utility functions for CIL.

In this repository, we provide the following utility classes:
- [AverageMeter.py](https://github.com/GregoirePetit/utilsCIL/blob/main/AverageMeter.py): Computes and stores the average and current value. Imported from pytorch, [here](https://github.com/pytorch/examples/blob/master/imagenet/main.py#L420-L438)
- [MyImageFolder.py](https://github.com/GregoirePetit/utilsCIL/blob/main/MyImageFolder.py): Custom ImageFolder class to load images from a list in [this format](https://github.com/GregoirePetit/imagelistsCIL). Example of usage in [this file](https://github.com/GregoirePetit/FeTrIL/blob/main/codes/scratch.py#L88-L104). Then it can be loaded in a dataloader with the following code:
```python
from MyImageFolder import ImagesListFileFolder
train_dataset = ImagesListFileFolder(train_list, transform=transform_train, random_seed=args.random_seed, range_classes=args.range_classes)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True, num_workers=args.workers,
    pin_memory=True)
```
- [MyFeatureFolder.py](https://github.com/GregoirePetit/utilsCIL/blob/main/MyFeatureFolder.py): Custom ImageFolder class to load features. Example of usage in [this file](https://github.com/GregoirePetit/FeTrIL/blob/main/codes/train_classifiers.py#L40). Then it can be loaded in a dataloader with the following code:
```python
from MyFeatureFolder import L4FeaturesDataset
train_dataset = L4FeaturesDataset(train_dir, range_classes=args.range_classes)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True, num_workers=args.workers,
    pin_memory=True)
```
- [Utils.py](https://github.com/GregoirePetit/utilsCIL/blob/main/Utils.py): Utility functions for CIL.
- [modified_resnet.py]((https://github.com/GregoirePetit/utilsCIL/blob/main/modified_resnet.py): Custom ResNet model with different architectures, from [Learning a Unified Classifier Incrementally via Rebalancing](https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/imagenet-class-incremental/modified_resnet.py)
- [modified_linear.py]((https://github.com/GregoirePetit/utilsCIL/blob/main/modified_linear.py): Custom linear model with different architectures, from [Learning a Unified Classifier Incrementally via Rebalancing](https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/imagenet-class-incremental/modified_linear.py)