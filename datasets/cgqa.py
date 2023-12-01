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

import numpy as np
import torch
from torch.utils.data.dataset import Subset, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from einops.layers.torch import Rearrange

from avalanche.benchmarks.classic.classic_benchmarks_utils import check_vision_benchmark
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.generators import nc_benchmark, dataset_benchmark
from avalanche.benchmarks.utils import PathsDataset, \
    classification_subset, concat_classification_datasets, make_classification_dataset


""" README
The original labels of classes are the sorted combination of all existing
objects defined in json. E.g., "apple,banana".
"""


class RandomGridPosition(object):
    """Random permutation of grid positions. Need to be after transforms.ToTensor()"""
    def __init__(self, grid_size=(2, 2)):
        self.grid_size = grid_size

    def __call__(self, img):
        """
        Args:
            img: Torch image
        Returns:
            Torch image
        """
        to_grids = Rearrange('c (g1 ph) (g2 pw) -> (g1 g2) c ph pw', g1=self.grid_size[0], g2=self.grid_size[1])
        to_img = Rearrange('(g1 g2) c ph pw -> c (g1 ph) (g2 pw)', g1=self.grid_size[0], g2=self.grid_size[1])
        img_grids = to_grids(img)       # [g1*g2, c, h/g1, w/g1]

        num_grids = img_grids.shape[0]
        perm_img_grids = img_grids[torch.randperm(num_grids)]
        perm_img = to_img(perm_img_grids)

        return perm_img


def _build_default_transform(image_size=(128, 228), is_train=True, normalize=True):
    """
    Default transforms borrowed from MetaShift.
    Imagenet normalization.
    """
    _train_transform = [
            transforms.Resize(image_size),  # allow reshape but not equal scaling
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # RandomGridPosition(),
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
        multi_task: bool = False,
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
    :param multi_task: if True, return a multi_task benchmark,
        else, return a continual learning benchmark.

    :returns: A properly initialized instance: `GenericCLScenario`
        with train_stream, val_stream, test_stream.
    """
    if dataset_root is None:
        dataset_root = default_dataset_location("CFST")

    if train_transform is None:
        train_transform = _build_default_transform(image_size, True)
    if eval_transform is None:
        eval_transform = _build_default_transform(image_size, False)

    '''load datasets'''
    if num_samples_each_label is None or num_samples_each_label < 0:
        num_samples_each_label = None

    datasets, label_info = _get_gqa_datasets(dataset_root, mode='continual', image_size=image_size,
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

    def obtain_subset(dataset, exp_idx, memory_size=0):
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

        mapped_targets = np.array([class_mappings[exp_idx][idx] for idx in np.array(t)[indices]])
        return make_classification_dataset(
            MySubset(dataset, indices=list(indices), class_mapping=class_mappings[exp_idx]),
            targets=mapped_targets,
            task_labels=task_labels,
        )

    train_subsets = [
        obtain_subset(train_set, expidx, memory_size)
        for expidx in range(n_experiences)
    ]

    val_subsets = [
        obtain_subset(val_set, expidx)
        for expidx in range(n_experiences)
    ]

    test_subsets = [
        obtain_subset(test_set, expidx)
        for expidx in range(n_experiences)
    ]

    if multi_task:
        train_subsets = [
            concat_classification_datasets(
                train_subsets,
                # transform_groups={'val': (None, None)},
        )]

    benchmark_instance = dataset_benchmark(
        train_datasets=train_subsets,
        test_datasets=test_subsets,
        other_streams_datasets={'val': val_subsets},
        train_transform=train_transform,
        eval_transform=eval_transform,
        other_streams_transforms={'val': (eval_transform, None)},
    )
    benchmark_instance.original_classes_in_exp = original_classes_in_exp
    benchmark_instance.classes_in_exp = classes_in_exp
    benchmark_instance.class_mappings = class_mappings
    benchmark_instance.n_classes = num_classes
    benchmark_instance.label_info = label_info
    benchmark_instance.return_task_id = return_task_id

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
    :param mode: Option [sys, pro, sub, non, noc].
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
        dataset_root = default_dataset_location("CFST")

    if train_transform is None:
        train_transform = _build_default_transform(image_size, True)
    if eval_transform is None:
        eval_transform = _build_default_transform(image_size, False)

    '''load datasets'''
    datasets, label_info = _get_gqa_datasets(dataset_root, mode=mode, image_size=image_size)
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
            classification_subset(
                dataset,
                indices=shot_indices,
                class_mapping=class_mappings[exp_idx],
                transform_groups={'val': (None, None)},
                task_labels=task_labels[exp_idx])
        )
        val_subsets.append(
            classification_subset(
                dataset,
                indices=val_indices,
                class_mapping=class_mappings[exp_idx],
                transform_groups={'val': (None, None)},
                task_labels=task_labels[exp_idx])
        )
        test_subsets.append(
            classification_subset(
                dataset,
                indices=query_indices,
                class_mapping=class_mappings[exp_idx],
                transform_groups={'val': (None, None)},
                task_labels=task_labels[exp_idx])
        )

    benchmark_instance = dataset_benchmark(
        train_datasets=train_subsets,
        test_datasets=test_subsets,
        other_streams_datasets={'val': val_subsets},
        train_transform=train_transform,
        eval_transform=eval_transform,
        other_streams_transforms={'val': (eval_transform, None)},
    )

    benchmark_instance.original_classes_in_exp = np.array(selected_classes_in_exp)
    benchmark_instance.classes_in_exp = np.array(classes_in_exp)
    benchmark_instance.class_mappings = np.array(class_mappings)
    benchmark_instance.n_classes = num_classes
    benchmark_instance.label_info = label_info
    benchmark_instance.return_task_id = True

    return benchmark_instance


def _get_gqa_datasets(
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
    :param mode: Option [continual, sys, pro, sub, non, noc].
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
    img_folder_path = os.path.join(dataset_root, "CGQA", "GQA_100")

    def preprocess_label_to_integer(img_info, mapping_tuple_label_to_int):
        for item in img_info:
            if item['newImageName'].startswith('task'):     # TaskEvaluation needs to remove path prefix
                item['newImageName'] = '/'.join(item['newImageName'].split('/')[1:])
            item['image'] = f"{item['newImageName']}.jpg"
            item['label'] = mapping_tuple_label_to_int[tuple(sorted(item['comb']))]
            for obj in item['objects']:
                obj['image'] = f"{obj['imageName']}.jpg"

    def formulate_img_tuples(images):
        """generate train_list and test_list: list with img tuple (path, label)"""
        img_tuples = []
        for item in images:
            instance_tuple = (item['image'], item['label'])     # , item['boundingBox']
            img_tuples.append(instance_tuple)
        return img_tuples

    if mode == 'continual':
        train_json_path = os.path.join(img_folder_path, "continual", "train", "train.json")
        val_json_path = os.path.join(img_folder_path, "continual", "val", "val.json")
        test_json_path = os.path.join(img_folder_path, "continual", "test", "test.json")

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
        label_set = sorted(list(set([tuple(sorted(item['comb'])) for item in val_img_info])))
        # [('building', 'sign'), ...]
        map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
        # {('building', 'sign'): 0, ('building', 'sky'): 1, ...}
        map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
        # {0: ('building', 'sign'), 1: ('building', 'sky'),...}

        preprocess_label_to_integer(train_img_info, map_tuple_label_to_int)
        preprocess_label_to_integer(val_img_info, map_tuple_label_to_int)
        preprocess_label_to_integer(test_img_info, map_tuple_label_to_int)

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

    else:
        if mode in ['sys', 'pro', 'sub', 'non', 'noc']:
            json_name = {'sys': 'sys/sys_fewshot.json', 'pro': 'pro/pro_fewshot.json', 'sub': 'sub/sub_fewshot.json',
                         'non': 'non_novel/non_novel_fewshot.json', 'noc': 'non_comp/non_comp_fewshot.json'}[mode]
        else:
            raise Exception(f'Un-implemented mode "{mode}".')
        json_path = os.path.join(img_folder_path, "fewshot", json_name)
        with open(json_path, 'r') as f:
            img_info = json.load(f)
        label_set = sorted(list(set([tuple(sorted(item['comb'])) for item in img_info])))
        map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
        map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
        preprocess_label_to_integer(img_info, map_tuple_label_to_int)
        img_list = formulate_img_tuples(img_info)
        dataset = PathsDataset(
            root=img_folder_path,
            files=img_list,
            transform=transforms.Compose([transforms.Resize(image_size)]))

        datasets = {'dataset': dataset}
        label_info = (label_set, map_tuple_label_to_int, map_int_label_to_tuple)

    return datasets, label_info


class MySubset(Subset):
    """
    subset with class mapping
    """
    def __init__(self, dataset, indices: list, class_mapping, transform=None):
        super().__init__(dataset, indices)
        # self._dataset = dataset
        # self._indices = indices
        # self._subset = Subset(dataset, indices)
        self._class_mapping = class_mapping
        self._transform = transform

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        if self._transform is not None:
            x = self._transform(x)
        mapped_y = self._class_mapping[y]
        return x, mapped_y


__all__ = ["continual_training_benchmark", "fewshot_testing_benchmark", "build_transform_for_vit"]
