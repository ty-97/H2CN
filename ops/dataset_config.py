# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

"""-------------------------------------------------------------------------------------"""
"""
### Sport_Project
### CUSTOM_DATASET
"""
""" BASKETBALL """
ROOT_DATASET = "/data1/yanbin/"
ROOT_DATASET2 = "/home/vip/Datasets/"

def return_egaze1(modality):
    filename_categories = 106
    # filename_categories = 'soccer_allframe256_v2j/soccer_classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET2
        filename_imglist_train = root_data + 'EGTEA_Gaze+/trainValTest/train1.txt'
        filename_imglist_val = root_data + 'EGTEA_Gaze+/trainValTest/val1.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_egaze2(modality):
    filename_categories = 106
    # filename_categories = 'soccer_allframe256_v2j/soccer_classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET2
        filename_imglist_train = root_data + 'EGTEA_Gaze+/trainValTest/train2.txt'
        filename_imglist_val = root_data + 'EGTEA_Gaze+/trainValTest/val2.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_egaze3(modality):
    filename_categories = 106
    # filename_categories = 'soccer_allframe256_v2j/soccer_classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET2
        filename_imglist_train = root_data + 'EGTEA_Gaze+/trainValTest/train3.txt'
        filename_imglist_val = root_data + 'EGTEA_Gaze+/trainValTest/val3.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_diving48(modality):
    filename_categories = 48
    # filename_categories = 'soccer_allframe256_v2j/soccer_classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET
        filename_imglist_train = root_data + 'diving48/trainValTest/train.txt'
        filename_imglist_val = root_data + 'diving48/trainValTest/val.txt'
        prefix = '{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_basketball(modality):
    # filename_categories = 'basketball_allframe256_v2j/basketball_classInd.txt'
    filename_categories = 8
    if modality == 'RGB':
        root_data = ROOT_DATASET

        filename_imglist_train = '/vireo00/yanbin2/Sport_Video/videos_our/TraValTes/basketball_tsm/train.txt'
        filename_imglist_val   = '/vireo00/yanbin2/Sport_Video/videos_our/TraValTes/basketball_tsm/val.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_soccer(modality):
    filename_categories = 10
    # filename_categories = 'soccer_allframe256_v2j/soccer_classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + frameFolder1
        filename_imglist_train = '/vireo00/yanbin2/Sport_Video/videos_our/TraValTes/soccer_tsm/train.txt'
        filename_imglist_val = '/vireo00/yanbin2/Sport_Video/videos_our/TraValTes/soccer_tsm/val.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
"""-------------------------------------------------------------------------------------"""


def return_ucf101(modality):
    # filename_categories = 'UCF101/labels/classInd.txt'
    filename_categories = 101
    if modality == 'RGB':
        root_data = "/vireo00/yanbin2/Sport_Video/UCF"

        filename_imglist_train = '/vireo00/yanbin2/Sport_Video/UCF/traval_tsm/train.txt'
        filename_imglist_val   = '/vireo00/yanbin2/Sport_Video/UCF/traval_tsm/val.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_rgb_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv1(modality):
    # filename_categories = 'something/v1/category.txt'
    filename_categories = 174
    if modality == 'RGB':
        root_data = '/data/tanyi/'
        filename_imglist_train = root_data+'some_some_v1/trainValTest/train.txt'
        filename_imglist_val = root_data+'some_some_v1/trainValTest/val.txt'
        prefix = '{:05d}.jpg'
    # elif modality == 'Flow':
    #     root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-flow'
    #     filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
    #     filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
    #     prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    # filename_categories = 'something/v2/category.txt'
    filename_categories = 174
    if modality == 'RGB':
        root_data = ROOT_DATASET
        filename_imglist_train = root_data+ 'some_some_v2/trainValTest/train.txt'
        filename_imglist_val = root_data+ 'some_some_v2/trainValTest/val.txt'
        prefix = '{:06d}.jpg'
    # elif modality == 'Flow':
    #     root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
    #     filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
    #     filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
    #     prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ROOT_DATASET + 'jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET
        filename_imglist_train = root_data + 'kinetics400_mmlab/trainValTest/train.txt'
        filename_imglist_val = root_data + 'kinetics400_mmlab/trainValTest/val.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kineticsmini1(modality):
    filename_categories = 100
    if modality == 'RGB':
        root_data = ROOT_DATASET
        filename_imglist_train = '/data/vireodata/video_data/kinetics400/mini_kinetics_100/train.txt'
        filename_imglist_val = '/data/vireodata/video_data/kinetics400/mini_kinetics_100/val.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_kineticsmini2(modality):
    filename_categories = 200
    if modality == 'RGB':
        root_data = ROOT_DATASET
        filename_imglist_train = '/data/vireodata/video_data/kinetics400/mini-kinetics-200/train.txt'
        filename_imglist_val = '/data/vireodata/video_data/kinetics400/mini-kinetics-200/val.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester, 'somethingv1': return_somethingv1, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                   'kinetics': return_kinetics, 'kineticsmini1': return_kineticsmini1, 'kineticsmini2': return_kineticsmini2,
                   ### CUSTOM_DATASET
                   'basketball': return_basketball, 'soccer': return_soccer, 'diving48': return_diving48,
                   'egaze1': return_egaze1, 'egaze2': return_egaze2, 'egaze3': return_egaze3
                   }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    # file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    # file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
