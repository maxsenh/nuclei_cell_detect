# Tutorial on how to train with the given framework

There are several important steps one has to do.

1. Pre-training models

The models given by maskrcnn-benchmark or detectron were trained with 81 classes, therefore one needs to remove the last layers of the pretrained model using the file `parsing/trim_detectron_model.py` like this `python parsing/trim_detectron_model.py --pretrained_path ... --save_path ... --cfg ...` where one can specify the keys to be removed within the file like this:
```python
newdict['model'] = removekey(_d['model'],
                                ['mask_fcn_logits.bias', 'mask_fcn_logits.weight', 'cls_score.weight', 'cls_score.bias', 'bbox_pred.weight', 'bbox_pred.bias'])
```

2. Augmentation

There is a script provided, which does offline augmentation during handling of the annotation file called New_chunk.ipynb. Online augmentation is implemented, but not throughouly tested.

3. Training

Training was done starting from a labelbox annotation file. There are scripts provided to parse them, chunk them in the right size and augment them. Currently, only the data set class from COCO is used to for data handling, but this will be updated soon.

4. Finetuning of model

The idea is to run inference on a trained model and then use the images, which have not performed well, label them and retrain the model again.
