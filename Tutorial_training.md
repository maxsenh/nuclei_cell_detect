# Tutorial on how to train with the given framework

There are several important steps one has to do.

1. Install

Install nuclei_cell_detect according to INSTALL.md. Help can be found in the original Mask R-CNN benchmark repository.

2. Create data set

Data set creation is done with the script `parsing/create_dataset.ipynb`. Here, an annotation file from LabelBox is handled. In the next steps, the images are downloaded, chunked and transformed into the COCO format. After chunking the images into numpy arrays, the arrays have to be separated into training, validation and testing data. Then, each of the three datasets is created into COCO-format, which includes for each a folder with the images and a annotation file.

3. Configure maskrcnn_benchmark/config/paths_catalog.py

The paths of the image data and the annotation data have to be added to the maskrcnn-benchmark/maskrcnn_benchmark/config/paths_catalog.py file similar to this:

    "coco_nuclei_train": { 
            "img_dir": "/data/proj/smFISH/Students/Max_Senftleben/files/data/nuclei_20190205_data/train",
            "ann_file":	"/data/proj/smFISH/Students/Max_Senftleben/files/annotation/new_nuclei/train_coco_id.json"
        },
    "coco_nuclei_val": { 
            "img_dir": "/data/proj/smFISH/Students/Max_Senftleben/files/data/nuclei_20190205_data/val",
            "ann_file":	"/data/proj/smFISH/Students/Max_Senftleben/files/annotation/new_nuclei/val_coco_id.json"
        },
    "coco_nuclei_test": { 
            "img_dir": "/data/proj/smFISH/Students/Max_Senftleben/files/data/nuclei_20190205_data/test",
            "ann_file":	"/data/proj/smFISH/Students/Max_Senftleben/files/annotation/new_nuclei/test_coco_id.json"
        }
        
4. Configure existing or make own .yaml configuration file

Existing config files can be found in configs/, datasets have to be specified at DATASETS according to the keywords in the paths_catalog.py file:

    DATASETS:
      TRAIN: ("coco_nuclei_train", "coco_nuclei_val")
      TEST: ("coco_nuclei_test",)

Here, several training parameters may be specified and adjusted to the number of GPUs being used while training. These parameters are for training on ONE GPU:

    SOLVER:
      BASE_LR: 0.0025
      STEPS: (480000, 640000)
      MAX_ITER: 720000
      IMS_PER_BATCH: 2

The developer's recommmend multiplying the learning rate and the images per batch by the number of GPUs and dividing the steps and the maximum iterations by the number GPUs. See examples in the original repository maskrcnn-benchmark/configs/, where they were training with 8 GPUs. One may add the output directory as well with `OUTPUT_DIR: "/path/to/"`.

I ran into an error `IndexError: index 0 is out of bounds for dimension 0 with size 0` while training on a test data set from labelbox with relatively small .jpeg images (I did not get the error while training the nuclei though). A quick fix can be setting `DATALOADER.ASPECT_RATIO_GROUPING = FALSE` in the config file, but the developpers are still checking the code and have not provided another solution.

5. Pre-training models

The models given by maskrcnn-benchmark or detectron were trained with 81 classes, therefore one needs to remove the last layers of the pretrained model using the file `parsing/trim_detectron_model.py` by writing bash:
`python parsing/trim_detectron_model.py --pretrained_path <old_path> --save_path <trimmed_path> --cfg <path>` where one can specify the keys to be removed (hardcoded) in the file:
```python
newdict['model'] = removekey(_d['model'],
                                ['mask_fcn_logits.bias', 'mask_fcn_logits.weight', 'cls_score.weight', 'cls_score.bias', 'bbox_pred.weight', 'bbox_pred.bias'])
```

The pretraining model has to be updated in the configuration file under MODEL.WEIGHT and the maskrcnn_benchmark/config/paths_catalog.py.

6. Training

Run training by writing in the terminal `python maskrcnn-benchmark/tools/train_net.py --config-file "path/to/config.yaml"`. When receiving Segmentation fault error, the GCC version may be updated (has to be version +4.9). Info here https://paper.dropbox.com/doc/Working-on-Monod-setup-environments-and-run-on-GPUs--AXH64wJuBgEe8XwCtJ09DZCqAg-hX2FfDYdlhY10ksm0BhH6. Checking the GCC version cna be done with `gcc --version`. After updating GCC to a higher version one has to recompile maskrcnn-benchmark by removing the folder `maskrcnn-benchmark/build` and do `python maskrcnn-benchmark/setup.py build develop`. The loss together with the inference over the iterations can be estimated with `parsing/get_loss_new.ipynb`.

7. Inference

Inference, or the comparison between predicted and ground truth data, can be done with the `demo/Inference_pdf.ipynb` notebook. If scores need to be obtained, run `python tools/test_net.py --config-file configs/<config>'`. In order to obtain the IoU and AP 0.5 for every intermediate model that is saved, one can run this command with a bash for-loop, save the output to a .txt file and parse the file with `parsing/get_loss_new.ipynb`.

8. Prediction of test data

See `demo/nuclei_cell_demo.ipynb`. This is mainly a demo for simple prediction tasks. In the notebook `demo/predict_folder.ipynb`, however, a whole folder can be predicted. There, scripts are provided with which the predictions of the test images can be transformed into numpy array files and can further be chunked. These chunks can then in the next step be created as a data set together with the dataset creation script from above. In the folder `parsing/`, scripts can be found to estimate the average instance size/area. 

9. Re-training or transfer learning

With the use of new data, either from step 8 or simply data that, additionally, has been manually, can be used to re-train models. To do that, refer to one of the transfer learning configuration files. The pre-trained model would then preferably be a model that has been trained before on nuclei/cells. With `load_state_dict.ipynb` the keys optimizer, scheduler and iteration can be removed, which would interfer with the re-training process. 


