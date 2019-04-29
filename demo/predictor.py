# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torchvision import transforms as T

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util

# added imports
from maskrcnn_benchmark.utils.imports import import_file
import json
from pycocotools.coco import *
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from maskrcnn_benchmark.config import cfg
import matplotlib.patches as patches
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from matplotlib.patches import Rectangle, Polygon
from PIL import Image
from matplotlib.ticker import NullLocator
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 20,12

class NUCLEIdemo(object):
    # My categories
    CATEGORIES = [
        "__background",
        "nuclei",
        "undefined",
        "clusters"
    ]

    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)        
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)

        return result, predictions
    
    
    def get_ax(self, rows=1, cols=1, size=16):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.

        Adjust the size attribute to control how big to render images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax
    
    def on_test_images(self, image_folder, result_folder = None, add_class_names = None):
        predis = []

        for image in os.listdir(image_folder):
            
            # transform image to PIL
            pil_img = Image.open(image_folder + image)
            
            
            img = np.array(pil_img)[:, :, [0, 1, 2]]
            #img = np.dstack((img, img, img))
            print(img.shape)
            print(img.dtype)
            print(img)
            
            # compute predictions
            predictions = self.compute_prediction(img)
            predis.append(predictions)
            top_predictions = self.select_top_predictions(predictions)
            
            result = img.copy()
            
            if self.show_mask_heatmaps:
                return self.create_mask_montage(result, top_predictions)
            
            result = self.overlay_boxes(result, top_predictions)
            if add_class_names:
                result = self.overlay_class_names(result, top_predictions)
            if self.cfg.MODEL.MASK_ON:
                result = self.overlay_mask(result, top_predictions)
            
            if result_folder:
                
                # show prediction
                ax = self.get_ax()
                ax.imshow(result)
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                            hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
        
                plt.savefig(result_folder + image[:-4] + '_pred.png', bbox_inches = 'tight', pad_inches = 0)
                plt.close()
            else:
                plt.imshow(Image.fromarray(result))
                plt.show()
        return predis
                
                
    def inference(self, add_class_names = None, save_path = None, save_independently = None):
        """
        Do Inference, either show the boxes or the masks
        """
        
        paths_catalog = import_file("maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
        )
        DatasetCatalog = paths_catalog.DatasetCatalog
        test_datasets = DatasetCatalog.get(cfg.DATASETS.TEST[0])
        img_dir = test_datasets['args']['root']
        anno_file = test_datasets['args']['ann_file']
        data = json.load(open(anno_file))
        coco = COCO(anno_file)
        predis = []
        filenames = []
        
        # iterate through data
        for i, image in enumerate(data['images']):
            
            pil_img = Image.open(img_dir + '/' + image['file_name'])
            filenames.append(image['file_name'])
            img = np.array(pil_img)[:, :, [0, 1, 2]]
            # get ground truth boxes or masks
            anno = [obj for obj in data['annotations'] if obj['image_id'] == image['id']]
            classes = [obj['category_id'] for obj in data['annotations'] if obj['image_id'] == image['id']]
            json_category_id_to_contiguous_id = {
                    v: i + 1 for i, v in enumerate(coco.getCatIds())
            }
            classes = [json_category_id_to_contiguous_id[c] for c in classes]
            classes = torch.tensor(classes)
            boxes = [obj['bbox'] for obj in anno]
            boxes = torch.as_tensor(boxes).reshape(-1,4)
            target = BoxList(boxes, pil_img.size, mode = 'xywh').convert('xyxy')
            target.add_field('labels', classes)
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size)
            target.add_field("masks", masks)
            target = target.clip_to_image(remove_empty=True)
            
            polygons = []
            color_rgb = [[255,0,0], [255, 0, 113], [255, 87, 0], [255,101,80]]
            colors = {i: [s/255 for s in color] for i, color in enumerate(color_rgb)}
            color = [colors[i.item()] for i in classes]
            boxes = []
    
            polys = vars(target)['extra_fields']['masks']
            for polygon in polys:
                try:
                    tenso = vars(polygon)['polygons'][0]
                except KeyError:
                    continue
                poly1 = tenso.numpy()
                poly = poly1.reshape((int(len(poly1)/2),2))
                polygons.append(Polygon(poly))
        
            xywh_tar = target.convert("xywh")
            for box in vars(xywh_tar)['bbox'].numpy():

                rect = Rectangle((box[0],box[1]), box[2], box[3])
                boxes.append(rect)
            

            
            # show ground truth image in first plot
            fig = plt.figure()
            ax1 = fig.add_subplot(1,3,1)
            ax1.imshow(pil_img)
            plt.axis('off')
            
            # compute predictions
            predictions = self.compute_prediction(img)
            predis.append(predictions)
            top_predictions = self.select_top_predictions(predictions)
            result = img.copy()
            
            result_boxes = self.overlay_boxes(result, top_predictions)
            if add_class_names:
                result_boxes = self.overlay_class_names(result_boxes, top_predictions)
            result = img.copy()
            result_masks = self.overlay_mask(result, top_predictions)
            if add_class_names:
                result_masks = self.overlay_class_names(result_masks, top_predictions)

   
            if save_independently:
                # 1
                fig = plt.figure(dpi=150)
                ax = fig.add_subplot(1,1,1)
                
                result = img.copy()
                
                plt.imshow(Image.fromarray(result))
                plt.axis('off')
                
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
                plt.margins(0,0)
                
                plt.savefig(save_independently + image['file_name'][:-4] + '_trut.png',
                           bbox = "tight", pad_inches = 0)
                plt.show()
                
                
                # 2
                fig = plt.figure(dpi=150)
                ax = fig.add_subplot(1,1,1)
                plt.imshow(Image.fromarray(result_boxes))
                plt.axis('off')
                b = PatchCollection(boxes, facecolor = 'none', 
                                    linewidths = 2, edgecolor = color)
                ax.add_collection(b)
                
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
                plt.margins(0,0)
                
                plt.savefig(save_independently + image['file_name'][:-4] + '_bbox.png',
                           bbox = "tight", pad_inches = 0)
                plt.show()
                plt.close()
                
                # 3
                fig = plt.figure(dpi=150)
                ax = fig.add_subplot(1,1,1)
                plt.imshow(Image.fromarray(result_masks))
                plt.axis('off')
            
                p = PatchCollection(polygons, facecolor = 'none', linewidths = 0, alpha = 0.4)
                ax.add_collection(p)
                p = PatchCollection(polygons, facecolor = 'none', edgecolors = color, linewidths = 2)
                ax.add_collection(p)
                
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.savefig(save_independently + image['file_name'][:-4] + '_mask.png',
                           bbox = "tight", pad_inches = 0)
                plt.show()
                plt.close()
                
            # show boxes in second plot

            ax2 = fig.add_subplot(1,3,2)
            plt.imshow(Image.fromarray(result_boxes))
            plt.axis('off')
            b = PatchCollection(boxes, facecolor = 'none', linewidths = 2, edgecolor = color)
            ax2.add_collection(b)
            
            # show masks in third plot

            ax3 = fig.add_subplot(1,3,3)
            plt.imshow(Image.fromarray(result_masks))
            plt.axis('off')
            
            p = PatchCollection(polygons, facecolor = 'none', linewidths = 0, alpha = 0.4)
            ax3.add_collection(p)
            p = PatchCollection(polygons, facecolor = 'none', edgecolors = color, linewidths = 2)
            ax3.add_collection(p)
            
            if save_path:
                path = save_path + image['file_name'][:-4] + '_inference.png'
                plt.savefig(path)
                plt.show
            plt.show()
        dic = {}
        for i in range(len(filenames)):
            dic[filenames[i]] = predis[i]
        return dic
            

            
    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(np.uint8(original_image))
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 3
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            # fix for CV2 version check issue #339
            thresh = mask[0, :, :, None]
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # change width of the mask here
            image = cv2.drawContours(image, contours, -1, color, 2)

        composite = image

        return composite

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        
        labels = predictions.get_field("labels").tolist()
        

        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox
        width, heigth, dim = image.shape
        font_scale = (width * heigth) / (1000 * 1000)
        font_scale = 1
              
        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)

            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2
            )

        return image
