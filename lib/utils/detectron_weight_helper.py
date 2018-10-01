"""Helper functions for loading pretrained weights from Detectron pickle files
"""

import pickle
import re
import torch


def xfer_skip(name):
    return name.startswith("cls_score") \
           or name.startswith("mask_fcn_logits") \
           or name.startswith("bbox_pred")


def load_detectron_weight(net, detectron_weight_file, xfer=False, freeze=False):
    if xfer and freeze:
        print("Freezing weights")
    elif freeze:
        raise ValueError("Freeze only means something with xfer")
    elif xfer:
        print("Not freezing weights")
    name_mapping, orphan_in_detectron = net.detectron_weight_mapping

    with open(detectron_weight_file, 'rb') as fp:
        src_blobs = pickle.load(fp, encoding='latin1')
    if 'blobs' in src_blobs:
        src_blobs = src_blobs['blobs']

    params = net.state_dict()
    for p_name, p_tensor in params.items():
        d_name = name_mapping[p_name]
        if isinstance(d_name, str):  # maybe str, None or True
            # i.e., copy unless you're doing xfer and the name is blacklisted
            if not (xfer and xfer_skip(d_name)):
                p_tensor.copy_(torch.Tensor(src_blobs[d_name]))
                if xfer and freeze:
                    p_tensor.requires_grad_ = False


def resnet_weights_name_pattern():
    pattern = re.compile(r"conv1_w|conv1_gn_[sb]|res_conv1_.+|res\d+_\d+_.+")
    return pattern


if __name__ == '__main__':
    """Testing"""
    from pprint import pprint
    import sys
    sys.path.insert(0, '..')
    from modeling.model_builder import Generalized_RCNN
    from core.config import cfg, cfg_from_file

    cfg.MODEL.NUM_CLASSES = 81
    cfg_from_file('../../cfgs/res50_mask.yml')
    net = Generalized_RCNN()

    # pprint(list(net.state_dict().keys()), width=1)

    mapping, orphans = net.detectron_weight_mapping
    state_dict = net.state_dict()

    for k in mapping.keys():
        assert k in state_dict, '%s' % k

    rest = set(state_dict.keys()) - set(mapping.keys())
    assert len(rest) == 0
