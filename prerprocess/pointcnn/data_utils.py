from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import h5py
import plyfile
import random
import numpy as np
from matplotlib import cm


def save_ply(points, filename, colors=None, normals=None):
    vertex = np.array([tuple(p) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    n = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = np.array([tuple(n) for n in normals], dtype=[('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
        assert len(vertex_normal) == n
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        vertex_color = np.array([tuple(c * 255) for c in colors],
                                dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        assert len(vertex_color) == n
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(n, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def save_ply_property(points, property, property_max, filename, cmap_name='tab20'):
    point_num = points.shape[0]
    colors = np.full(points.shape, 0.5)
    cmap = cm.get_cmap(cmap_name)
    for point_idx in range(point_num):
        if property[point_idx] == 0:
            colors[point_idx] = np.array([0, 0, 0])
        else:
            colors[point_idx] = cmap(property[point_idx] / property_max)[:3]
    save_ply(points, filename, colors)


def save_ply_batch(points_batch, file_path, points_num=None):
    batch_size = points_batch.shape[0]
    if type(file_path) != list:
        basename = os.path.splitext(file_path)[0]
        ext = '.ply'
    for batch_idx in range(batch_size):
        point_num = points_batch.shape[1] if points_num is None else points_num[batch_idx]
        if type(file_path) == list:
            save_ply(points_batch[batch_idx][:point_num], file_path[batch_idx])
        else:
            save_ply(points_batch[batch_idx][:point_num], '%s_%04d%s' % (basename, batch_idx, ext))


def save_ply_color_batch(points_batch, colors_batch, file_path, points_num=None):
    batch_size = points_batch.shape[0]
    if type(file_path) != list:
        basename = os.path.splitext(file_path)[0]
        ext = '.ply'
    for batch_idx in range(batch_size):
        point_num = points_batch.shape[1] if points_num is None else points_num[batch_idx]
        if type(file_path) == list:
            save_ply(points_batch[batch_idx][:point_num], file_path[batch_idx], colors_batch[batch_idx][:point_num])
        else:
            save_ply(points_batch[batch_idx][:point_num], '%s_%04d%s' % (basename, batch_idx, ext),
                     colors_batch[batch_idx][:point_num])


def save_ply_property_batch(points_batch, property_batch, file_path, points_num=None, property_max=None,
                            cmap_name='tab20'):
    batch_size = points_batch.shape[0]
    if type(file_path) != list:
        basename = os.path.splitext(file_path)[0]
        ext = '.ply'
    property_max = np.max(property_batch) if property_max is None else property_max
    for batch_idx in range(batch_size):
        point_num = points_batch.shape[1] if points_num is None else points_num[batch_idx]
        if type(file_path) == list:
            save_ply_property(points_batch[batch_idx][:point_num], property_batch[batch_idx][:point_num],
                              property_max, file_path[batch_idx], cmap_name)
        else:
            save_ply_property(points_batch[batch_idx][:point_num], property_batch[batch_idx][:point_num],
                              property_max, '%s_%04d%s' % (basename, batch_idx, ext), cmap_name)


def save_ply_point_with_normal(data_sample, folder):
    for idx, sample in enumerate(data_sample):
        filename_pts = os.path.join(folder, '{:08d}.ply'.format(idx))
        save_ply(sample[..., :3], filename_pts, normals=sample[..., 3:])


def grouped_shuffle(inputs):
    for idx in range(len(inputs) - 1):
        assert (len(inputs[idx]) == len(inputs[idx + 1]))

    shuffle_indices = np.arange(len(inputs[0]))
    np.random.shuffle(shuffle_indices)
    outputs = []
    for idx in range(len(inputs)):
        for i in range(len(inputs[0])):
            outputs.append(inputs[idx][shuffle_indices[i], ...])
        outputs.append(inputs[idx][shuffle_indices[i], ...])
    return outputs


def load_cls(filelist):
    points = []
    labels = []

    folder = os.path.dirname(filelist)
    for line in open(filelist):
        filename = os.path.basename(line.rstrip())
        data = h5py.File(os.path.join(folder, filename))
        if 'normal' in data:
            points.append(np.concatenate([data['data'][...], data['normal'][...]], axis=-1).astype(np.float32))
        else:
            points.append(data['data'][...].astype(np.float32))
        labels.append(np.squeeze(data['label'][:]).astype(np.int64))
    return (np.concatenate(points, axis=0),
            np.concatenate(labels, axis=0))


def load_cls_train_val(filelist, filelist_val):
    data_train, label_train = grouped_shuffle(load_cls(filelist))
    data_val, label_val = load_cls(filelist_val)
    return data_train, label_train, data_val, label_val


def is_h5_list(filelist):
    return all([line.strip()[-3:] == '.h5' for line in open(filelist)])


def load_seg_list(filelist):
    folder = os.path.dirname(filelist)
    return [os.path.join(folder, line.strip()) for line in open(filelist)]


def load_seg(filelist):
    points = []
    labels = []
    point_nums = []
    labels_seg = []
    indices_split_to_full = []

    folder = os.path.dirname(filelist)
    for line in open(filelist):
        data = h5py.File(os.path.join(folder, line.strip()))
        points.append(data['data'][...].astype(np.float32))
        labels.append(data['label'][...].astype(np.int64))
        point_nums.append(data['data_num'][...].astype(np.int32))
        labels_seg.append(data['label_seg'][...].astype(np.int64))
        if 'indices_split_to_full' in data:
            indices_split_to_full.append(data['indices_split_to_full'][...].astype(np.int64))

    return (np.concatenate(points, axis=0),
            np.concatenate(labels, axis=0),
            np.concatenate(point_nums, axis=0),
            np.concatenate(labels_seg, axis=0),
            np.concatenate(indices_split_to_full, axis=0) if indices_split_to_full else None)


def balance_classes(labels):
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    counts_max = np.amax(counts)
    repeat_num_avg_unique = counts_max / counts
    repeat_num_avg = repeat_num_avg_unique[inverse]
    repeat_num_floor = np.floor(repeat_num_avg)
    repeat_num_probs = repeat_num_avg - repeat_num_floor
    repeat_num = repeat_num_floor + (np.random.rand(repeat_num_probs.shape[0]) < repeat_num_probs)

    return repeat_num.astype(np.int64)

def load_cls_pts(filelist,k):

    labels = []
    points = []
    for i in range(k):
        folder = os.path.dirname(filelist)
        for line in open(filelist):
            filename , label = line.split(" ",1)
            _,filename = filename.split("./")
            file_path = os.path.join(folder, filename)
            data = []
            count = 0
            with open(file_path) as fpts:
                while 1:
                    line = fpts.readline()
                    if not line:
                        break;
                    L = line.split(' ')
                    L = [float(i) for i in L]
                    data.append(np.array(L))
                    count = count + 1
                if(count>=2048):
                    index = random.sample(range(0,count),2048)
                    pointcloud = np.zeros((2048, 6),dtype=np.float32)
                    for i in range(2048):
                        pointcloud[i] = data[index[i]]
                    points.append(pointcloud)
                    labels.append(int(label))

    labels = np.array(labels)
    points = np.array(points)
    return points , labels


def load_ptsfile(filelist,k=1):
    data_train, label_train = grouped_shuffle(load_cls_pts(filelist,k))
    return data_train, label_train

def load_file(filelist):

    labels = []
    weights = []
    points_path = []
    sizes = []
    len = 0
    for line in open(filelist):
        size = np.zeros((1,3))
        filename , label, weight, size[0,0], size[0,1], size[0,2] = line.split(" ")
        points_path.append(filename)
        weights.append(float(weight))
        labels.append(int(label))
        sizes.append(size)
        #print(size[0,0], size[0,1], size[0,2])
        len = len +1
    labels = np.array(labels)
    return points_path, labels, weights, sizes, len

def read_path(filelist):

    read_paths = []
    write_paths = []
    len = 0
    for line in open(filelist):
        read_path, write_path = line.split(" ")
        write_path = write_path.rstrip('\n')
        read_paths.append(read_path)
        write_paths.append(write_path)
        len = len +1
    return read_paths, write_paths, len

def read_test_path(filelist):

    read_paths = []
    sizes = []
    obbs = []
    gts = []
    write_paths = []
    feature_paths = []
    len = 0
    for line in open(filelist):
        size = np.zeros((1,3))
        obb = np.zeros(8)
        read_path, obb[0], obb[1], obb[2], obb[3], obb[4], obb[5], obb[6], obb[7], gt, write_path, feature_path = line.split(" ")
        size[0,0], size[0,1], size[0,2] = obb[0], obb[1], obb[2]
        feature_path = feature_path.rstrip('\n')
        read_paths.append(read_path)
        sizes.append(size)
        obbs.append(obb)
        gts.append(int(gt))
        write_paths.append(write_path)
        feature_paths.append(feature_path)
        len = len +1
    return read_paths, sizes, obbs, gts, write_paths, feature_paths, len