"""
Modified from SpiderCNN: https://github.com/xyf513/SpiderCNN
Author: Jiachen Xu and Jingyu Gong
Date: June 2020
"""
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR#os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'shapenet'))
import provider
import tf_util
import part_dataset_all_normal


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='scene_encoder_rsl_shapenet', help='Model name [default: model]')
parser.add_argument('--log_dir', default='visualize', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=16881*20, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
#os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
#os.system('cp train_GPU1.py %s' % (LOG_DIR)) # bkp of train procedure
#LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
#LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 50

# Shapenet official train/test split
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
TRAIN_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='trainval', return_cls_label=True)
TEST_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='test', return_cls_label=True)

def log_string(out_str):
    #LOG_FOUT.write(out_str+'\n')
    #LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def visualize_all():
    num_votes = 1
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            #pointclouds_pl, labels_pl, cls_labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            pointclouds_pl, labels_pl, labels_onehot_pl, cls_labels_pl, external_scene_encode_pl, cos_loss_weight = MODEL.placeholder_scene_inputs(BATCH_SIZE, NUM_POINT,NUM_CLASSES)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            #print is_training_pl

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            #print "--- Get model and loss"
            # Get model and loss
            #pred = MODEL.get_model(pointclouds_pl, cls_labels_pl, is_training_pl, bn_decay=bn_decay, num_classes=NUM_CLASSES)
            pred_origin, end_points, external_scene_feature = MODEL.get_scene_model(pointclouds_pl, cls_labels_pl, is_training_pl, bn_decay=bn_decay, num_classes=NUM_CLASSES)
            #loss = MODEL.get_loss(pred, labels_pl)
            loss, pred, loss_decomposed = MODEL.get_scene_loss(cos_loss_weight, pred_origin, labels_pl, labels_onehot_pl, external_scene_feature, external_scene_encode_pl, end_points['feats'], pointclouds_pl[:, :, 0:3])
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            #print "--- Get training operator"
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        #train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        #test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        #sess.run(init)
        #sess.run(init, {is_training_pl: True})
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'labels_onehot_pl': labels_onehot_pl,
               'cls_labels_pl': cls_labels_pl,
               'is_training_pl': is_training_pl,
               'external_scene_encode_pl': external_scene_encode_pl,
               'pred': pred,
               'loss': loss,
               'loss_decomposed': loss_decomposed,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'cos_loss_weight': cos_loss_weight}
        eval_scene_one_epoch(sess, ops, NUM_CLASSES)

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_cls_label = np.zeros((bsize,), dtype=np.int32)
    for i in range(bsize):
        ps,normal,seg,cls = dataset[idxs[i+start_idx]]
        batch_data[i,:,0:3] = ps
        batch_data[i,:,3:6] = normal
        batch_label[i,:] = seg
        batch_cls_label[i] = cls
    return batch_data, batch_label, batch_cls_label

def create_color_palette():
    return [
       (255, 0, 0),
       (174, 199, 232),
       (152, 223, 138),
       (31, 119, 180),
       (255, 187, 120),
       (188, 189, 34),
       (140, 86, 75),
       (255, 152, 150),
       (214, 39, 40),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),
       (247, 182, 210),
       (66, 188, 102),
       (219, 219, 141),
       (140, 57, 197),
       (202, 185, 52),
       (51, 176, 203),
       (200, 54, 131),
       (92, 193, 61),
       (78, 71, 183),
       (172, 114, 82),
       (255, 127, 14),
       (91, 163, 138),
       (153, 98, 156),
       (140, 153, 101),
       (158, 218, 229),
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),
       (112, 128, 144),
       (96, 207, 209),
       (227, 119, 194),
       (213, 92, 176),
       (94, 106, 211),
       (82, 84, 163),
       (100, 85, 144),
       (0, 85, 14),
       (120, 18, 28),
       (46, 211, 14),
       (144, 120, 24),
       (122, 228, 34),
       (196, 107, 129),
       (127, 129, 94),
       (113, 192, 126),
       (194, 126, 121),
       (62, 184, 63)
    ]

def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')
    ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
            \n
            '''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)
    return

def visualize_instance(xyz, pred, output_file):
    if not output_file.endswith('.ply'):
        print('output file must be a .ply file')
        exit(0)
    colors = create_color_palette()
    num_colors = len(colors)
    ids = pred
    vertex_color = np.zeros((xyz.shape[0], 3), dtype=np.int32)
    for i in range(xyz.shape[0]):
        if ids[i] >= num_colors:
                print('found predicted label ' + str(ids[i]) + ' not in nyu40 label set')
                exit()
        color = colors[ids[i]]
        vertex_color[i,0] = color[0]
        vertex_color[i,1] = color[1]
        vertex_color[i,2] = color[2]
    create_output(xyz, vertex_color, output_file)
    return


def eval_scene_one_epoch(sess, ops, num_classes):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    instance_index = 0
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (len(TEST_DATASET)+BATCH_SIZE-1)/BATCH_SIZE
    num_batches = int(num_batches)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    seg_classes = TEST_DATASET.seg_classes
    shape_ious = {cat:[] for cat in seg_classes.keys()}
    seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT)).astype(np.int32)
    batch_cls_label = np.zeros((BATCH_SIZE,)).astype(np.int32)
    for batch_idx in range(num_batches):
        if batch_idx %20==0:
            log_string('%03d/%03d'%(batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        cur_batch_data, cur_batch_label, cur_batch_cls_label = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)
        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_label = cur_batch_label
            batch_cls_label = cur_batch_cls_label
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_label[0:cur_batch_size] = cur_batch_label
            batch_cls_label[0:cur_batch_size] = cur_batch_cls_label
        batch_label_onehot = np.eye(num_classes)[batch_label]
        external_batch_scene_encode = np.max(batch_label_onehot,axis=1)

        # ---------------------------------------------------------------------
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['labels_onehot_pl']: batch_label_onehot,
                     ops['cls_labels_pl']: batch_cls_label,
                     ops['external_scene_encode_pl']: external_batch_scene_encode,
                     ops['is_training_pl']: is_training,
                     ops['cos_loss_weight']: 1.0}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        # ---------------------------------------------------------------------

        # Select valid data
        cur_pred_val = pred_val[0:cur_batch_size]
        # Constrain pred to the groundtruth classes (selected by seg_classes[cat])
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        for i in range(cur_batch_size):
            cat = seg_label_to_cat[cur_batch_label[i,0]]
            logits = cur_pred_val_logits[i,:,:]
            cur_pred_val[i,:] = np.argmax(logits[:,seg_classes[cat]], 1) + seg_classes[cat][0]
        correct = np.sum(cur_pred_val == cur_batch_label)
        total_correct += correct
        total_seen += (cur_batch_size*NUM_POINT)
        if cur_batch_size==BATCH_SIZE:
            loss_sum += loss_val
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(cur_batch_label==l)
            total_correct_class[l] += (np.sum((cur_pred_val==l) & (cur_batch_label==l)))

        for i in range(cur_batch_size):
            segp = cur_pred_val[i,:]
            segl = cur_batch_label[i,:]
            seg_xyz = batch_data[i,:,0:3]
            cat = seg_label_to_cat[segl[0]]
            tmp_path = os.path.join(LOG_DIR, cat)
            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)
            predict_output_file = os.path.join(tmp_path, "%04d_predict.ply"%instance_index)
            gt_output_file = os.path.join(tmp_path, "%04d_gt.ply"%instance_index)
            visualize_instance(seg_xyz, segp, predict_output_file)
            visualize_instance(seg_xyz, segl, gt_output_file)
            instance_index += 1

    EPOCH_CNT += 1
    return


if __name__ == "__main__":
    #log_string('pid: %s'%(str(os.getpid())))
    visualize_all()
    #train()
    #LOG_FOUT.close()
