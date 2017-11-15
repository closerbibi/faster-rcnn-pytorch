# Adapted the following lib/datasets/inria.py for py-faster-rcnn
# https://github.com/zeyuanxy/fast-rcnn/blob/master/lib/datasets/inria.py

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
#import datasets.inria
#from .inria import inria
import os
#from datasets.imdb import imdb
from .imdb import imdb
#import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
from utils import cython_bbox
import cPickle
import subprocess
import pdb
import uuid
from voc_eval import voc_eval

#class inria(datasets.imdb.imdb):
class inria(imdb):
    def __init__(self, image_set, devkit_path):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data', 'DIRE')
        self._classes = ('__background__','jacket', 'traffic cone', 'pen', 'antenna', 'fireplug', 'bird',
       'shirt', 'screen door', 'laptop', 'banner', 'videos', 'washer',
       'backpack', 'post', 'toilet tissue', 'teapot', 'bar', 'knife',
       'grill', 'coffee maker', 'radiator', 'bulletin board', 'place mat',
       'drawing', 'land', 'document', 'booklet', 'pitcher', 'pipe',
       'blanket', 'oven', 'dishwasher', 'central reservation', 'board',
       'beam', 'pane', 'step', 'hood', 'loudspeaker', 'cup', 'bridge',
       'snow', 'candle', 'curb', 'doorframe', 'bus', 'hat', 'river',
       'hill', 'animal', 'bucket', 'container', 'microwave', 'manhole',
       'text', 'mug', 'counter', 'countertop', 'sand', 'double door',
       'wardrobe', 'apparel', 'napkin', 'ottoman', 'can', 'refrigerator',
       'magazine', 'fan', 'toilet', 'bathtub', 'bookcase', 'umbrella',
       'food', 'sculpture', 'fireplace', 'screen', 'telephone', 'computer',
       'monitor', 'bicycle', 'air conditioner', 'fruit', 'trade name',
       'stairway', 'bannister', 'tray', 'figurine', 'stove', 'blind',
       'poster', 'field', 'truck', 'path', 'base', 'ball', 'switch',
       'minibike', 'television receiver', 'chest of drawers', 'paper',
       'water', 'fluorescent', 'boat', 'chandelier', 'shoe', 'sea',
       'candlestick', 'flag', 'van', 'stool', 'clock', 'ashcan', 'awning',
       'bowl', 'coffee table', 'stairs', 'jar', 'skyscraper', 'basket',
       'bench', 'house', 'swivel chair', 'palm', 'traffic light',
       'plaything', 'towel', 'railing', 'work surface', 'desk', 'seat',
       'bag', 'column', 'rug', 'sink', 'glass', 'wall socket', 'mirror',
       'sofa', 'plate', 'sconce', 'spotlight', 'flower', 'fence', 'pole',
       'vase', 'shelf', 'earth', 'rock', 'pillow', 'armchair', 'bed',
       'bottle', 'mountain', 'pot', 'box', 'grass', 'door', 'streetlight',
       'curtain', 'book', 'cushion', 'road', 'sidewalk', 'signboard',
       'lamp', 'ceiling', 'cabinet', 'table', 'sky', 'painting',
       'windowpane', 'light', 'plant', 'floor', 'chair', 'car', 'building',
       'tree', 'person', 'wall')
        self._wnid = range(181)
        '''
        self._classes = ('__background__', # always index 0
                         'partition', 'jacket', 'traffic cone', 'pen', 'antenna', 'fireplug',
       'rack', 'statue', 'embankment', 'soap dispenser', 'bird', 'pier',
       'kitchen island', 'shirt', 'screen door', 'laptop', 'banner',
       'binder', 'teacup', 'airplane', 'videos', 'washer', 'backpack',
       'post', 'toilet tissue', 'case', 'teapot', 'bar', 'knife', 'grill',
       'coffee maker', 'radiator', 'gate', 'bulletin board',
       'parking meter', 'arcade machine', 'place mat', 'drawing', 'land',
       'gym shoe', 'faucet', 'document', 'notebook', 'booklet', 'pitcher',
       'pipe', 'blanket', 'oven', 'dishwasher', 'hedge',
       'central reservation', 'board', 'beam', 'pane', 'step', 'hood',
       'loudspeaker', 'cup', 'bridge', 'snow', 'candle', 'curb',
       'doorframe', 'bus', 'hat', 'river', 'hill', 'animal', 'bucket',
       'container', 'microwave', 'manhole', 'text', 'mug', 'counter',
       'countertop', 'sand', 'double door', 'wardrobe', 'apparel',
       'napkin', 'ottoman', 'can', 'refrigerator', 'magazine', 'fan',
       'toilet', 'bathtub', 'bookcase', 'gravestone', 'umbrella', 'food',
       'sculpture', 'fireplace', 'screen', 'telephone', 'computer',
       'monitor', 'bicycle', 'air conditioner', 'fruit', 'trade name',
       'stairway', 'bannister', 'tray', 'figurine', 'stove', 'blind',
       'poster', 'field', 'truck', 'path', 'base', 'ball', 'switch',
       'minibike', 'television receiver', 'chest of drawers', 'paper',
       'water', 'fluorescent', 'boat', 'chandelier', 'shoe', 'sea',
       'candlestick', 'flag', 'van', 'stool', 'clock', 'ashcan', 'awning',
       'shrub', 'bowl', 'coffee table', 'stairs', 'jar', 'skyscraper',
       'basket', 'bench', 'house', 'swivel chair', 'palm', 'traffic light',
       'plaything', 'towel', 'railing', 'work surface', 'desk', 'seat',
       'bag', 'column', 'rug', 'sink', 'glass', 'wall socket', 'mirror',
       'sofa', 'plate', 'sconce', 'spotlight', 'flower', 'fence', 'pole',
       'vase', 'shelf', 'earth', 'rock', 'pillow', 'armchair', 'bed',
       'bottle', 'mountain', 'pot', 'box', 'grass', 'door', 'streetlight',
       'curtain', 'book', 'cushion', 'road', 'sidewalk', 'signboard',
       'lamp', 'ceiling', 'cabinet', 'table', 'sky', 'painting',
       'windowpane', 'light', 'plant', 'floor', 'chair', 'car', 'building',
       'tree', 'person', 'wall')
        self._wnid = range(201)
        '''
        '''
        self._classes = ('__background__', # always index 0
                         'chair')
        self._wnid = (0,5)
        '''
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._wnid_to_ind = dict(zip(self._wnid, xrange(self.num_classes)))
        self._image_ext = ['.jpg'] #npy
        self._image_index = self._load_image_set_index()
        #pdb.set_trace()
        # Default to roidb handler
        #self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        #self._salt = str(uuid.uuid4())
        self._salt = str(os.getpid())
        self._comp_id = 'comp4'
        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000,#,
                       'matlab_eval' : False,
                       'use_diff' : False,
                       'rpn_file' : None}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path,'Images',
                                  index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
	return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path,'ImageSets', 
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        gt_roidb = [self._load_inria_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        #pdb.set_trace()
        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
            print len(roidb)
	with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        gt_roidb = self.gt_roidb()
        #pdb.set_trace()
        #rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)
        #roidb = self._load_rpn_roidb(None)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        #pdb.set_trace()
        filename = self.config['rpn_file']
        #pdb.set_trace()
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            #pdb.set_trace()
            box_list = cPickle.load(f)
        #pdb.set_trace()
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self._devkit_path,
                                                self.name + '.npy'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
	raw_data = sio.loadmat(filename)['all_boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

	return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        eturn the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 self.name))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.npy')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_inria_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of INRIAPerson.
        """
        index=index.split('_')[0]+'_'+index.split('_')[1]+'_'+index.split('_')[2]
        filename = os.path.join(self._data_path, 'Annotations', index + '.txt')
        # print 'Loading: {}'.format(filename)
	with open(filename) as f:
            data = f.read()
	import re
	#objs = re.findall('\(\d+, \d+\)[\s\-]+\(\d+, \d+\)', data)
	objs = re.findall('\(\d+, \d+\)[\s\-]+\(\d+, \d+\)[\s\-]+\(\w+\)', data)

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
	    coor = re.findall('\d+', obj)
            x1 = float(coor[0])
            y1 = float(coor[1])
            x2 = float(coor[2])
            y2 = float(coor[3])
            #cls = self._class_to_ind['person']
            cls = self._class_to_ind[re.findall('\w+', obj)[-1]]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _write_inria_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '_{}'.format(os.getpid())

        # VOCdevkit/results/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', self.name, comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'setenv(\'LC_ALL\',\'C\'); voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir)
	pdb.set_trace()
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    """
    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_inria_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)
    """
    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_inria_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'test',
            filename)
        return path

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'data',
            'DIRE',
            'Annotations',
            '{:s}.txt')
        imagesetfile = os.path.join(
            self._devkit_path,
            'data',
            'DIRE',
            'ImageSets',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.inria('train', '')
    pdb.set_trace()
    res = d.roidb
    from IPython import embed; embed()

