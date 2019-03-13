# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
import random

def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        
        # if not, width and height will not work for the rotation
        if cfg.INPUT.ONLINE_AUGMENT == True:
            # 8 cases
            if cfg.INPUT.HEIGHT_IS_WIDTH:
                hflip_prob = 0.5
                vflip_prob = 0.5
                rot_prob = 1
            else:
                hflip_prob = 0.5
                vflip_prob = 0
                rot_prob = 0
        else:
            hflip_prob = 0
            rot_prob = 0
            vflip_prob = 0

    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        hflip_prob = 0
        rot_prob = 0
        vflip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    
    # own
    rotate_transform = T.Rotate(rot_prob)
    veri_flip = T.VerticalFlip(vflip_prob) 
    hori_flip = T.RandomHorizontalFlip(hflip_prob)
    
    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            hori_flip,
            veri_flip,
            rotate_transform,
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
