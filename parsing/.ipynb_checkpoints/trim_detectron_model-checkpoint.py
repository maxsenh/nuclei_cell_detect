import os
import torch
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    default="~/.torch/models/_detectron_35858933_12_2017_baselines_e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC_output_train_coco_2014_train%3Acoco_2014_valminusminival_generalized_rcnn_model_final.pkl",
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="./pretrained_model/mask_rcnn_R-50-FPN_1x_detectron_no_last_layers.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    default="configs/e2e_mask_rcnn_R_50_FPN_1x.yaml",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "--show_keys",
    default = None,
    help = 'show the keys present in the model',
    type=str,
)

args = parser.parse_args()
#
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

cfg.merge_from_file(args.cfg)
_d = load_c2_format(cfg, DETECTRON_PATH)
newdict = _d


newdict['model'] = removekey(_d['model'],
                                ['mask_fcn_logits.bias', 'mask_fcn_logits.weight', 'cls_score.weight', 'cls_score.bias', 'bbox_pred.weight', 'bbox_pred.bias'])

#newdict['model'] = removekey(_d['model'], [])

if args.show_keys == "True":
    print(_d['model'].keys())
    print("eys")
else:
    torch.save(newdict, args.save_path)
    print('saved to {}.'.format(args.save_path))
