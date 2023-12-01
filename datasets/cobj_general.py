################################################################################
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 17-10-2022                                                             #
# Author(s): Weiduo Liao                                                       #
# E-mail: liaowd@mail.sustech.edu.cn                                           #
################################################################################
import os
from pathlib import Path
from typing import Optional, Sequence, Union, Any, Dict, List
import json
from PIL import Image
from timm.data import create_transform

import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import crop, InterpolationMode
import torch
from torch import Tensor


""" README
The original labels of classes are the sorted combination of all existing
objects defined in json. E.g., "apple,banana".
"""


def _build_default_transform(image_size=(128, 228), is_train=True, normalize=True):
    """
    Default transforms borrowed from MetaShift.
    Imagenet normalization.
    """
    _train_transform = [
            transforms.Resize(image_size),  # allow reshape but not equal scaling
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
    ]
    _eval_transform = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
    ]
    if normalize:
        _train_transform.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ))
        _eval_transform.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ))

    _default_train_transform = transforms.Compose(_train_transform)
    _default_eval_transform = transforms.Compose(_eval_transform)

    if is_train:
        return _default_train_transform
    else:
        return _default_eval_transform


def build_transform_for_vit(img_size=(224, 224), is_train=True):
    if is_train:
        _train_transform = create_transform(
            input_size=img_size,
            is_training=is_train,
            color_jitter=0.3,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        # replace RandomResizedCropAndInterpolation with Resize, for not cropping img and missing concepts
        _train_transform.transforms[0] = transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC)

        return _train_transform
    else:
        return _build_default_transform(img_size, False)


def continual_training_benchmark(
        n_experiences: int,
        *,
        image_size=(128, 128),
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        train_transform: Optional[Any] = None,
        eval_transform: Optional[Any] = None,
        dataset_root: Union[str, Path] = None,
        memory_size: int = 0,
        num_samples_each_label: Optional[int] = None,
):
    """
    Creates a CL benchmark using the pre-processed GQA dataset.

    :param n_experiences: The number of experiences in the current benchmark.
    :param image_size: size of image.
    :param return_task_id: If True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to True.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset.
        Defaults to None, which means that the default location for
        'tinyimagenet' will be used.
    :param memory_size: Total memory size for store all past classes/tasks.
        Each class has equal number of instances in the memory.
    :param num_samples_each_label: Number of samples for each label,
        -1 or None means all data are used.

    :returns: A properly initialized instance: `GenericCLScenario`
        with train_stream, val_stream, test_stream.
    """
    if dataset_root is None:
        dataset_root = './cobj'

    if train_transform is None:
        train_transform = _build_default_transform(image_size, True)
    if eval_transform is None:
        eval_transform = _build_default_transform(image_size, False)

    '''load datasets'''
    if num_samples_each_label is None or num_samples_each_label < 0:
        num_samples_each_label = None

    datasets, label_info = _get_obj365_datasets(dataset_root, mode='continual', image_size=image_size,
                                                num_samples_each_label=num_samples_each_label)
    train_set, val_set, test_set = datasets['train'], datasets['val'], datasets['test']
    label_set, map_tuple_label_to_int, map_int_label_to_tuple = label_info

    '''generate class order for continual tasks'''
    num_classes = len(label_set)
    assert num_classes % n_experiences == 0
    num_class_in_exp = num_classes // n_experiences
    classes_order = np.array(list(map_int_label_to_tuple.keys())).astype(np.int64)  # [0-99]
    if fixed_class_order is not None:
        assert len(fixed_class_order) == num_classes
        classes_order = np.array(fixed_class_order).astype(np.int64)
    elif shuffle:
        rng = np.random.RandomState(seed=seed)
        rng.shuffle(classes_order)

    original_classes_in_exp = classes_order.reshape([n_experiences, num_class_in_exp])  # e.g.[[5, 2], [6, 10],...]
    if return_task_id:      # task-IL
        classes_in_exp = np.stack([np.arange(num_class_in_exp) for _ in range(n_experiences)]).astype(np.int64)
        # [[0,1], [0,1],...]
    else:
        classes_in_exp = np.arange(num_classes).reshape([n_experiences, num_class_in_exp]).astype(np.int64)
        # [[0,1], [2,3],...]

    '''class mapping for each exp, contain the mapping for previous exps (unseen filled with -1)'''
    '''so that it allow memory buffer for previous exps'''
    class_mappings = []
    for exp_idx in range(n_experiences):
        class_mapping = np.array([-1] * num_classes)
        class_mapping[original_classes_in_exp[:exp_idx+1].reshape(-1)] = classes_in_exp[:exp_idx+1].reshape(-1)
        class_mappings.append(class_mapping)    # [-1 -1  2 ... -1  6 -1 ... -1  0 -1 ... -1]
    class_mappings = np.array(class_mappings).astype(np.int64)

    '''get sample indices for each experiment'''
    rng = np.random.RandomState(seed)   # reset rng for memory selection

    def obtain_subset(dataset, exp_idx, memory_size, transform=None):
        t = dataset.targets
        exp_classes = original_classes_in_exp[exp_idx]
        indices = [np.where(np.isin(t, exp_classes))[0]]    # current exp
        task_id = exp_idx if return_task_id else 0
        task_labels = [[task_id for _ in range(len(indices[0]))]]

        if memory_size > 0 and exp_idx > 0:
            old_classes = original_classes_in_exp[:exp_idx].reshape(-1)
            class_task_ids = {
                cls: t_id if return_task_id else 0
                for t_id, clses in enumerate(original_classes_in_exp[:exp_idx]) for cls in clses}

            num_instances_each_class = int(memory_size / len(old_classes))
            for cls in old_classes:
                cls_indices = np.where(t == cls)[0]
                rng.shuffle(cls_indices)
                indices.append(cls_indices[:num_instances_each_class])
                task_labels.append([class_task_ids[cls] for _ in range(len(indices[-1]))])

        indices = np.concatenate(indices)
        task_labels = np.concatenate(task_labels)
        assert indices.shape[0] == task_labels.shape[0]

        return Subset(
            dataset,
            indices=indices,
            class_mapping=class_mappings[exp_idx],
            task_labels=task_labels,
            transform=transform,
        )

    train_subsets = [
        obtain_subset(train_set, expidx, memory_size, train_transform)
        for expidx in range(n_experiences)
    ]

    val_subsets = [
        obtain_subset(val_set, expidx, 0, eval_transform)
        for expidx in range(n_experiences)
    ]

    test_subsets = [
        obtain_subset(test_set, expidx, 0, eval_transform)
        for expidx in range(n_experiences)
    ]

    benchmark_instance = Benchmark(
        train_datasets=train_subsets,
        test_datasets=test_subsets,
        val_datasets=val_subsets,
    )

    benchmark_instance.original_classes_in_exp = original_classes_in_exp
    benchmark_instance.classes_in_exp = classes_in_exp
    benchmark_instance.class_mappings = class_mappings
    benchmark_instance.n_classes = num_classes
    benchmark_instance.label_info = label_info
    benchmark_instance.x_dim = (3, *image_size)    # (3, 98, 98)
    benchmark_instance.n_experiences = n_experiences

    return benchmark_instance


def fewshot_testing_benchmark(
        n_experiences: int,
        *,
        image_size=(128, 128),
        n_way: int = 10,
        n_shot: int = 10,
        n_val: int = 5,
        n_query: int = 10,
        mode: str = 'sys',
        task_offset: int = 10,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        train_transform: Optional[Any] = None,
        eval_transform: Optional[Any] = None,
        dataset_root: Union[str, Path] = None,
):
    """
    Creates a CL benchmark using the pre-processed GQA dataset.

    For fewshot testing, you need to specify the specific testing mode.

    :param n_experiences: The number of experiences in the current benchmark.
        In the fewshot setting, it means the number of few-shot tasks
    :param image_size: size of image.
    :param n_way: Number of ways for few-shot tasks.
    :param n_shot: Number of support image instances for each class.
    :param n_val: Number of evaluation image instances for each class.
    :param n_query: Number of query image instances for each class.
    :param mode: Option [sys, pro, non, noc].
    :param task_offset: Offset for tasks not start from 0 in task-IL.
        Default to 10 since continual training consists of 10 tasks.
        You need to specify to 1 for class-IL.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset.
        Defaults to None, which means that the default location for
        'tinyimagenet' will be used.

    :returns: A properly initialized instance: `GenericCLScenario`.
    """
    if dataset_root is None:
        dataset_root = './cobj'

    if train_transform is None:
        train_transform = _build_default_transform(image_size, True)
    if eval_transform is None:
        eval_transform = _build_default_transform(image_size, False)

    '''load datasets'''
    datasets, label_info = _get_obj365_datasets(dataset_root, mode=mode, image_size=image_size)
    dataset = datasets['dataset']
    label_set, map_tuple_label_to_int, map_int_label_to_tuple = label_info

    '''generate fewshot tasks'''
    num_classes = len(label_set)
    classes_order = list(map_int_label_to_tuple.keys())  # [0-99]
    selected_classes_in_exp = []  # e.g.[[5, 4], [6, 5],...]
    classes_in_exp = []  # [[0,1], [0,1],...]
    class_mappings = []
    task_labels = []
    t = np.array(dataset.targets)
    train_subsets, val_subsets, test_subsets = [], [], []

    if fixed_class_order is not None:
        assert len(fixed_class_order) == n_experiences * n_way
        selected_classes_in_exp = np.array(fixed_class_order).astype(np.int64).reshape(n_experiences, n_way)
    else:
        rng = np.random.RandomState(seed=seed)
        for exp_idx in range(n_experiences):
            '''select n_way classes for each exp'''
            selected_class_idxs = rng.choice(classes_order, n_way, replace=False).astype(np.int64)
            selected_classes_in_exp.append(selected_class_idxs)

    for exp_idx in range(n_experiences):
        selected_class_idxs = selected_classes_in_exp[exp_idx]
        classes_in_exp.append(np.arange(n_way).astype(np.int64))
        class_mapping = np.array([-1] * num_classes)
        class_mapping[selected_class_idxs] = np.arange(n_way)
        class_mappings.append(class_mapping.astype(np.int64))
        task_labels.append(exp_idx + task_offset)

    rng = np.random.RandomState(seed)
    for exp_idx in range(n_experiences):
        selected_class_idxs = selected_classes_in_exp[exp_idx]
        '''select n_shot+n_val+n_query images for each class'''
        shot_indices, val_indices, query_indices = [], [], []
        for cls_idx in selected_class_idxs:
            indices = rng.choice(np.where(t == cls_idx)[0], n_shot + n_val + n_query, replace=False)
            shot_indices.append(indices[:n_shot])
            val_indices.append(indices[n_shot:n_shot+n_val])
            query_indices.append(indices[n_shot+n_val:])
        shot_indices = np.concatenate(shot_indices)
        if n_val > 0:
            val_indices = np.concatenate(val_indices)
        if n_query > 0:
            query_indices = np.concatenate(query_indices)
        train_subsets.append(
            Subset(
                dataset,
                indices=shot_indices,
                class_mapping=class_mappings[exp_idx],
                task_labels=task_labels[exp_idx],
                transform=train_transform,
            )
        )
        val_subsets.append(
            Subset(
                dataset,
                indices=val_indices,
                class_mapping=class_mappings[exp_idx],
                task_labels=task_labels[exp_idx],
                transform=eval_transform,
            )
        )
        test_subsets.append(
            Subset(
                dataset,
                indices=query_indices,
                class_mapping=class_mappings[exp_idx],
                task_labels=task_labels[exp_idx],
                transform=eval_transform,
            )
        )

    benchmark_instance = Benchmark(
        train_datasets=train_subsets,
        test_datasets=test_subsets,
        val_datasets=val_subsets,
    )

    benchmark_instance.original_classes_in_exp = np.array(selected_classes_in_exp)
    benchmark_instance.classes_in_exp = np.array(classes_in_exp)
    benchmark_instance.class_mappings = np.array(class_mappings)
    benchmark_instance.n_classes = num_classes
    benchmark_instance.label_info = label_info
    benchmark_instance.x_dim = (3, *image_size)    # (3, 98, 98)
    benchmark_instance.n_experiences = n_experiences

    return benchmark_instance


def _get_obj365_datasets(
        dataset_root,
        image_size=(128, 128),
        shuffle=False, seed: Optional[int] = None,
        mode='continual',
        num_samples_each_label=None,
        label_offset=0,
):
    """
    Create GQA dataset, with given json files,
    containing instance tuples with shape (img_name, label).

    You may need to specify label_offset if relative label do not start from 0.

    :param dataset_root: Path to the dataset root folder.
    :param image_size: size of image.
    :param shuffle: If true, the train sample order (in json)
        in the incremental experiences is
        randomly shuffled. Default to False.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param mode: Option [continual, sys, pro, sub, non, noc, nons, syss].
    :param num_samples_each_label: If specify a certain number of samples for each label,
        random sampling (build-in seed:1234,
        and replace=True if num_samples_each_label > num_samples, else False)
        is used to sample.
        Only for continual mode, only apply to train dataset.
    :param label_offset: specified if relative label not start from 0.
    :param preprocessed (DISABLED): True, just load preprocessed images,
        specified by newImageName,
        while False, construct new image by the defined object list.
        Default True.

    :return data_sets defined by json file and label information.
    """
    img_folder_path = os.path.join(dataset_root, "COBJ", "annotations")

    def preprocess_label_to_integer(img_info, mapping_tuple_label_to_int, prefix=''):
        for item in img_info:
            item['image'] = f"{prefix}{item['imageId']}.jpg"
            item['label'] = mapping_tuple_label_to_int[tuple(sorted(item['label']))]

    def formulate_img_tuples(images):
        """generate train_list and test_list: list with img tuple (path, label)"""
        img_tuples = []
        for item in images:
            instance_tuple = (item['image'], item['label'])     # , item['boundingBox']
            img_tuples.append(instance_tuple)
        return img_tuples

    if mode == 'continual':
        train_json_path = os.path.join(img_folder_path, "O365_continual_train_crop.json")
        val_json_path = os.path.join(img_folder_path, "O365_continual_val_crop.json")
        test_json_path = os.path.join(img_folder_path, "O365_continual_test_crop.json")

        with open(train_json_path, 'r') as f:
            train_img_info = json.load(f)
        with open(val_json_path, 'r') as f:
            val_img_info = json.load(f)
        with open(test_json_path, 'r') as f:
            test_img_info = json.load(f)
        # img_info:
        # [{'newImageName': 'continual/val/59767',
        #   'comb': ['hat', 'leaves'],
        #   'objects': [{'imageName': '2416370', 'objName': 'hat',
        #                'attributes': ['red'], 'boundingBox': [52, 289, 34, 45]},...]
        #   'position': [4, 1]},...]

        '''preprocess labels to integers'''
        label_set = sorted(list(set([tuple(sorted(item['label'])) for item in val_img_info])))
        # [('building', 'sign'), ...]
        map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
        # {('building', 'sign'): 0, ('building', 'sky'): 1, ...}
        map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
        # {0: ('building', 'sign'), 1: ('building', 'sky'),...}

        preprocess_label_to_integer(train_img_info, map_tuple_label_to_int, prefix='continual/train/')
        preprocess_label_to_integer(val_img_info, map_tuple_label_to_int, prefix='continual/val/')
        preprocess_label_to_integer(test_img_info, map_tuple_label_to_int, prefix='continual/test/')

        '''if num_samples_each_label provided, sample images to balance each class for train set'''
        selected_train_images = []
        if num_samples_each_label is not None and num_samples_each_label > 0:
            imgs_each_label = dict()
            for item in train_img_info:
                label = item['label']
                if label in imgs_each_label:
                    imgs_each_label[label].append(item)
                else:
                    imgs_each_label[label] = [item]
            build_in_seed = 1234
            build_in_rng = np.random.RandomState(seed=build_in_seed)
            for label, imgs in imgs_each_label.items():
                selected_idxs = build_in_rng.choice(
                    np.arange(len(imgs)), num_samples_each_label,
                    replace=True if num_samples_each_label > len(imgs) else False)
                for idx in selected_idxs:
                    selected_train_images.append(imgs[idx])
        else:
            selected_train_images = train_img_info

        '''generate train_list and test_list: list with img tuple (path, label)'''
        train_list = formulate_img_tuples(selected_train_images)
        val_list = formulate_img_tuples(val_img_info)
        test_list = formulate_img_tuples(test_img_info)
        # [('continual/val/59767.jpg', 0),...

        '''shuffle the train set'''
        if shuffle:
            rng = np.random.RandomState(seed=seed)
            order = np.arange(len(train_list))
            rng.shuffle(order)
            train_list = [train_list[idx] for idx in order]

        '''generate train_set and test_set using PathsDataset'''
        '''TBD: use TensorDataset if pre-loading in memory'''
        train_set = PathsDataset(
            root=img_folder_path,
            files=train_list,
            transform=transforms.Compose([transforms.Resize(image_size)]))
        val_set = PathsDataset(
            root=img_folder_path,
            files=val_list,
            transform=transforms.Compose([transforms.Resize(image_size)]))
        test_set = PathsDataset(
            root=img_folder_path,
            files=test_list,
            transform=transforms.Compose([transforms.Resize(image_size)]))

        datasets = {'train': train_set, 'val': val_set, 'test': test_set}
        label_info = (label_set, map_tuple_label_to_int, map_int_label_to_tuple)

    elif mode in ['sys', 'pro', 'non', 'noc']:   # no sub
        json_name = {'sys': 'O365_sys_fewshot_crop.json', 'pro': 'O365_pro_fewshot_crop.json',
                     'non': 'O365_non_fewshot_crop.json', 'noc': 'O365_noc_fewshot_crop.json'}[mode]
        json_path = os.path.join(img_folder_path, json_name)
        with open(json_path, 'r') as f:
            img_info = json.load(f)
        label_set = sorted(list(set([tuple(sorted(item['label'])) for item in img_info])))
        map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
        map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
        preprocess_label_to_integer(img_info, map_tuple_label_to_int, prefix=f'fewshot/{mode}/')
        img_list = formulate_img_tuples(img_info)
        dataset = PathsDataset(
            root=img_folder_path,
            files=img_list,
            transform=transforms.Compose([transforms.Resize(image_size)]))

        datasets = {'dataset': dataset}
        label_info = (label_set, map_tuple_label_to_int, map_int_label_to_tuple)

    else:
        raise Exception(f'Un-implemented mode "{mode}".')

    return datasets, label_info


def default_image_loader(path):
    """
    Sets the default image loader for the Pytorch Dataset.

    :param path: relative or absolute path of the file to load.

    :returns: Returns the image as a RGB PIL image.
    """
    return Image.open(path).convert("RGB")


class PathsDataset(torch.utils.data.Dataset):
    """
    This class extends the basic Pytorch Dataset class to handle list of paths
    as the main data source.
    """

    def __init__(
        self,
        root,
        files,
        transform=None,
        target_transform=None,
        loader=default_image_loader,
    ):
        """
        Creates a File Dataset from a list of files and labels.

        :param root: root path where the data to load are stored. May be None.
        :param files: list of tuples. Each tuple must contain two elements: the
            full path to the pattern and its class label. Optionally, the tuple
            may contain a third element describing the bounding box to use for
            cropping (top, left, height, width).
        :param transform: eventual transformation to add to the input data (x)
        :param target_transform: eventual transformation to add to the targets
            (y)
        :param loader: loader function to use (for the real data) given path.
        """

        if root is not None:
            root = Path(root)

        self.root: Optional[Path] = root
        self.imgs = files
        self.targets = [img_data[1] for img_data in self.imgs]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Returns next element in the dataset given the current index.

        :param index: index of the data to get.
        :return: loaded item.
        """

        img_description = self.imgs[index]
        impath = img_description[0]
        target = img_description[1]
        bbox = None
        if len(img_description) > 2:
            bbox = img_description[2]

        if self.root is not None:
            impath = self.root / impath
        img = self.loader(impath)

        # If a bounding box is provided, crop the image before passing it to
        # any user-defined transformation.
        if bbox is not None:
            if isinstance(bbox, Tensor):
                bbox = bbox.tolist()
            img = crop(img, *bbox)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """
        Returns the total number of elements in the dataset.

        :return: Total number of dataset items.
        """

        return len(self.imgs)


class Subset(torch.utils.data.dataset.Dataset):
    """
    subset with class mapping
    """
    def __init__(self, dataset, indices, class_mapping, task_labels, transform=None):
        self._dataset = dataset
        self._indices = indices
        self._subset = torch.utils.data.Subset(dataset, indices)
        self._class_mapping = class_mapping
        self._task_labels = task_labels
        self._transform = transform

    def __getitem__(self, index):
        x, y = self._subset[index]
        if self._transform is not None:
            x = self._transform(x)
        mapped_y = self._class_mapping[y]
        return x, mapped_y

    def __len__(self):
        return len(self._indices)

    def get_task_label(self, index):
        if type(self._task_labels) is int:
            return self._task_labels
        return self._task_labels[index]


class Benchmark:
    """
    Benchmark of all experiments
    """
    def __init__(self, train_datasets, test_datasets, val_datasets):
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.val_datasts = val_datasets


__all__ = ["continual_training_benchmark", "fewshot_testing_benchmark"]
