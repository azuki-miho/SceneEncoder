"""
Modified from PointConv: https://github.com/DylanWusee/pointconv
Author: Jiachen Xu and Jingyu Gong
Date: June 2020
"""
import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from PointConv import feature_encoding_layer, feature_decoding_layer

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl

def placeholder_scene_inputs(batch_size, num_point, num_classes):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    labels_onehot_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point,num_classes))
    cls_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    external_scene_encode_pl = tf.placeholder(tf.int32,shape=(batch_size,num_classes))

    cos_loss_weight = tf.placeholder(tf.float32, shape=None)
    return pointclouds_pl, labels_pl, labels_onehot_pl, cls_labels_pl, external_scene_encode_pl, cos_loss_weight

def get_model(point_cloud, is_training, num_class, sigma, bn_decay=None, weight_decay = None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = point_cloud

    # Feature encoding layers
    l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_points, npoint=1024, radius = 0.1, sigma = sigma, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')
    l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1_points, npoint=256, radius = 0.2, sigma = 2 * sigma, K=32, mlp=[64,64,128], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
    l3_xyz, l3_points = feature_encoding_layer(l2_xyz, l2_points, npoint=64, radius = 0.4, sigma = 4 * sigma, K=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3')
    l4_xyz, l4_points = feature_encoding_layer(l3_xyz, l3_points, npoint=36, radius = 0.8, sigma = 8 * sigma, K=32, mlp=[256,256,512], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4')

    # Feature decoding layers
    l3_points = feature_decoding_layer(l3_xyz, l4_xyz, l3_points, l4_points, 0.8, 8 * sigma, 16, [512,512], is_training, bn_decay, weight_decay, scope='fa_layer1')
    l2_points = feature_decoding_layer(l2_xyz, l3_xyz, l2_points, l3_points, 0.4, 4 * sigma, 16, [256,256], is_training, bn_decay, weight_decay, scope='fa_layer2')
    l1_points = feature_decoding_layer(l1_xyz, l2_xyz, l1_points, l2_points, 0.2, 2 * sigma, 16, [256,128], is_training, bn_decay, weight_decay, scope='fa_layer3')
    l0_points = feature_decoding_layer(l0_xyz, l1_xyz, l0_points, l1_points, 0.1, sigma, 16, [128,128,128], is_training, bn_decay, weight_decay, scope='fa_layer4')

    # print('l0', l0_points.shape)

    # end_points['feats'] = l0_points

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, weight_decay=weight_decay, scope='fc2')

    return net, end_points

def get_scene_model(point_cloud, cls_label, is_training, bn_decay=None, num_classes = 50):
    point_cloud_with_norm = point_cloud
    point_cloud = point_cloud[:, :, 0:3]
    sigma = 0.05
    weight_decay = None
    num_class = num_classes
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = point_cloud_with_norm

    # Feature encoding layers
    l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_points, npoint=512, radius = 0.1, sigma = sigma, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')
    l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1_points, npoint=128, radius = 0.2, sigma = 2 * sigma, K=32, mlp=[64,64,128], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
    l3_xyz, l3_points = feature_encoding_layer(l2_xyz, l2_points, npoint=36, radius = 0.4, sigma = 4 * sigma, K=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3')
    l4_xyz, l4_points = feature_encoding_layer(l3_xyz, l3_points, npoint=16, radius = 0.8, sigma = 8 * sigma, K=8, mlp=[256,256,512], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4')
    external_l5_xyz,external_l5_points = feature_encoding_layer(l4_xyz, l4_points, npoint=8, radius = 1.6, sigma = 8 * sigma, K=8, mlp=[512,512,512], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='external_layer5')
    external_l6_scene_feature = tf.reduce_mean(external_l5_points,axis=1,keepdims=True)
    external_scene_feature = tf_util.dropout(external_l6_scene_feature, keep_prob=0.5, is_training=is_training, scope='external_dp')
    external_scene_feature = tf_util.conv1d(external_scene_feature, num_class, 1, padding='VALID', activation_fn=None, weight_decay=weight_decay, scope='external_fc')


    # Feature decoding layers
    l3_points = feature_decoding_layer(l3_xyz, l4_xyz, l3_points, l4_points, 0.8, 8 * sigma, 16, [512,512], is_training, bn_decay, weight_decay, scope='fa_layer1')
    l2_points = feature_decoding_layer(l2_xyz, l3_xyz, l2_points, l3_points, 0.4, 4 * sigma, 16, [256,256], is_training, bn_decay, weight_decay, scope='fa_layer2')
    l1_points = feature_decoding_layer(l1_xyz, l2_xyz, l1_points, l2_points, 0.2, 2 * sigma, 16, [256,128], is_training, bn_decay, weight_decay, scope='fa_layer3')
    l0_points = feature_decoding_layer(l0_xyz, l1_xyz, l0_points, l1_points, 0.1, sigma, 16, [128,128,128], is_training, bn_decay, weight_decay, scope='fa_layer4')

    # print('l0', l0_points.shape)

    # end_points['feats'] = l0_points

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, weight_decay=weight_decay, scope='fc2')

    return net, end_points, external_scene_feature

def get_mask(label1, label2):
    num_points1 = label1.get_shape().as_list()[1]
    num_points2 = label2.get_shape().as_list()[1]

    label1 = tf.expand_dims(label1, axis=-1)
    label2 = tf.expand_dims(label2, axis=-1)
    label2_transpose = tf.transpose(label2, perm=[0, 2, 1])

    label1 = tf.tile(label1, [1, 1, num_points2])
    label2_transpose = tf.tile(label2_transpose, [1, num_points1, 1])

    # if they have same label, the make value is -1
    mask = (tf.cast(tf.equal(label1, label2_transpose), tf.float32) - 1) * 255

    return mask


def knn1(xyz1, xyz2, k=4, label1=None, label2=None):
    xyz1 = tf.squeeze(xyz1)
    xyz2 = tf.squeeze(xyz2)

    xyz2_transpose = tf.transpose(xyz2, perm=[0, 2, 1])
    point_cloud_inner = tf.matmul(xyz1, xyz2_transpose)

    point_cloud_inner = -2 * point_cloud_inner
    xyz1_square = tf.reduce_sum(tf.square(xyz1), axis=-1, keep_dims=True)

    xyz2_square = tf.reduce_sum(tf.square(xyz2), axis=-1, keep_dims=True)

    xyz2_square_trans = tf.transpose(xyz2_square, perm=[0, 2, 1])


    adj_matrix = xyz1_square + point_cloud_inner + xyz2_square_trans

    neg_adj = -adj_matrix

    if label1 is not None:
        mask = get_mask(label1, label2)
        neg_adj += mask

    _, nn_idx = tf.nn.top_k(neg_adj, k=k)
    return nn_idx

def get_loss(cos_loss_weight, pred, label, label_onehot, external_scene_feature,external_scene_encode, point_features=None, xyz=None):
    """ pred: BxNxC,
        label: BxN,
	smpw: BxN """

    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred)
    pred = tf.nn.softmax(pred)
    #classify_loss = tf.keras.backend.categorical_crossentropy(label_onehot,pred)
    # classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    print("Here!")
    print(tf.get_collection('losses'))
    weight_reg = tf.add_n(tf.get_collection('losses'))
    classify_loss_mean = tf.reduce_mean(classify_loss, name='classify_loss_mean')
    total_loss = classify_loss_mean + weight_reg
    tf.summary.scalar('classify loss', classify_loss_mean)
    tf.summary.scalar('total loss', total_loss)

    return total_loss, pred, (classify_loss_mean, weight_reg)

def get_scene_loss(cos_loss_weight, pred, label, label_onehot, external_scene_feature,external_scene_encode, point_features=None, xyz=None):
    """ pred: BxNxC,
        label: BxN,
	smpw: BxN """
    if point_features is not None:
        pred_possibility = tf.reduce_max(tf.nn.softmax(pred, axis=-1), axis=-1)
        pred_class = tf.cast(tf.argmax(pred, axis=-1), tf.int32)

        batch_size, num_point, num_dims = point_features.get_shape()[:3]
        xyz_lst, feature_lst, label_lst = [], [], []

        def f1(correct_index1, i, possibility, correct_num):
            choice = tf.cond(tf.less(correct_num, 2048), lambda: f4(possibility, correct_num), lambda: f3(possibility))

            choice.set_shape([2048])

            correct_index = tf.squeeze(tf.gather(correct_index1, choice))

            xyz_lst.append(tf.expand_dims(tf.gather(xyz[i], correct_index), axis=0))
            feature_lst.append(tf.expand_dims(tf.gather(point_features[i], correct_index), axis=0))
            label_lst.append(tf.expand_dims(tf.gather(label[i], correct_index), axis=0))

            return tf.constant(False)

        def f2():
            loss_xyz = tf.concat(xyz_lst, axis=0)
            loss_feature = tf.concat(feature_lst, axis=0)
            loss_label = tf.concat(label_lst, axis=0)


            idx = knn1(loss_xyz, xyz, 8, loss_label, label)
            idx_ = tf.range(batch_size) * num_point
            idx_ = tf.reshape(idx_, [batch_size, 1, 1])

            point_features_flat = tf.reshape(point_features, [-1, num_dims])
            point_features_neighbors = tf.gather(point_features_flat, idx+idx_)
            point_features_central = tf.tile(tf.expand_dims(loss_feature, 2), [1, 1, 8, 1])

            point_features_central = tf.stop_gradient(point_features_central)

            norm1 = tf.sqrt(tf.reduce_sum(tf.square(point_features_neighbors), axis=-1))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(point_features_central), axis=-1))
            product = tf.reduce_sum(point_features_neighbors * point_features_central, axis=-1)
            cos_loss = tf.reduce_mean(product / (norm1 * norm2 + 1e-5))

            return 1 - cos_loss

        def f3(possibility):
            _, choice = tf.nn.top_k(possibility, k=2048)

            return choice

        def f4(possibility, correct_num):
            possibility_max = tf.argmax(possibility, output_type=tf.int32)

            _, choice = tf.nn.top_k(possibility, k=correct_num * 2 // 3)

            choice = tf.pad(choice, [[0, 2048 - correct_num * 2 // 3]], mode="CONSTANT", constant_values=possibility_max)

            return choice

        judge_lst = []
        for i in range(batch_size):
            index = tf.equal(pred_class[i], label[i])
            correct_num = tf.reduce_sum(tf.cast(index, tf.int32))
            correct_index1 = tf.reshape(tf.where(index), [-1])
            possibility = tf.gather(pred_possibility[i], correct_index1)

            judge_lst.append(tf.cond(tf.equal(correct_num, tf.constant(0)),
                                    lambda: tf.constant(True),
                                    lambda: f1(correct_index1, i, possibility, correct_num)))

        cos_loss = tf.cond(tf.equal(tf.reduce_sum(tf.cast(judge_lst, tf.int32)), tf.constant(0)), lambda: f2(), lambda: tf.constant(1.0))

    else:
        cos_loss = tf.constant(0.0)

    #classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred)
    external_scene_feature_2d = tf.squeeze(external_scene_feature,[1])
    external_scene_loss = tf.losses.sigmoid_cross_entropy(external_scene_encode,external_scene_feature_2d)
    external_scene_probability = tf.sigmoid(external_scene_feature)
    external_scene_probability = tf.tile(external_scene_probability,[1,pred.get_shape()[1],1])
    external_scene_probability = tf.stop_gradient(external_scene_probability)
    pred = tf.nn.softmax(pred)
    pred = tf.multiply(pred,external_scene_probability)
    softmax_probability_sum = tf.reduce_sum(pred,axis=2,keepdims=True)
    softmax_probability_sum = tf.tile(softmax_probability_sum,[1,1,pred.get_shape()[2]])
    pred = tf.div(pred,softmax_probability_sum+1e-5)
    classify_loss = tf.keras.backend.categorical_crossentropy(label_onehot,pred)
    # classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    print("Here!")
    print(tf.get_collection('losses'))
    #weight_reg = tf.add_n(tf.get_collection('losses'))
    classify_loss_mean = tf.reduce_mean(classify_loss, name='classify_loss_mean')
    total_loss = classify_loss_mean + cos_loss * cos_loss_weight + external_scene_loss
    tf.summary.scalar('classify loss', classify_loss_mean)
    tf.summary.scalar('total loss', total_loss)

    return total_loss, pred, (classify_loss_mean, cos_loss*cos_loss_weight, external_scene_loss)

if __name__=='__main__':
    import pdb
    pdb.set_trace()

    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10, 1.0)
        print(net)
