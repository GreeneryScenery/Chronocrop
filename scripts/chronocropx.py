# -*- coding: utf-8 -*-
"""ChronocropX.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CzsgS2o8-Sp3jGTZWAsyVm6eeG1yupxo

# Setup

## Settings
"""

import plotly.io as pio
import plotly.graph_objs as go
from prophet.plot import plot_plotly
from prophet import Prophet
import supervision as sv
import json
from shapely import geometry
import shutil
from torchvision import transforms
import torch
import cv2
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import math
import sys
import os
print('\u001b[37;1m {} \u001b[37;1m'.format('Settings'))

# @title Settings
WRITE: bool = True  # @param {type:'boolean'}
READ: bool = True  # @param {type:'boolean'}
CONTINUE: bool = False  # @param {type:'boolean'}
WRITE_IMAGE: bool = False  # @param {type:'boolean'}
READ_IMAGE: bool = True  # @param {type:'boolean'}
SAVE: bool = True  # @param {type:'boolean'}
PATH: str = 'SegmentPlants/inputs/8/aligned'  # @param {type:'string'}
# @param {type:'string'}
DATES_PATH: str = 'SegmentPlants/inputs/data/prediction/dates.csv'
# @param {type:'string'}
READ_ONLY_PATH: str = 'SegmentPlants/inputs/data/prediction/read_outputs.json'
# @param {type:'string'}
CONTINUE_WRITE_PATH: str = 'SegmentPlants/inputs/data/prediction/write_outputs.json'
# @param {type:'string'}
CONTINUE_READ_PATH: str = 'SegmentPlants/inputs/data/prediction/read_outputs.json'
CLASSES: list[str] = ['mango', 'romaine lettuce',
                      'tomato']  # @param {type:'raw'}
INDEX_TO_CLASS: dict[int, str] = {
    0: 'mango', 1: 'romaine', 2: 'tomato'}  # @param {type:'raw'}
INDEX_TO_SUBCLASS: dict[int, str] = {
    0: 'growing', 1: 'harvest', 2: 'ripe', 3: 'unripe'}  # @param {type:'raw'}
ARUCO_PX: int = 100  # @param {type:'integer'}
ARUCO_CM: int = 7  # @param {type:'integer'}
TARGET = 150  # @param {type:"integer"}
DAYS = 35  # @param {type:"integer"}
DISTANCE = 100  # @param {type:"integer"}
FAST: bool = True  # @param {type:'boolean'}

"""## Loading"""

print('\u001b[37;1m {} \u001b[37;1m'.format('Loading'))


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

colours = {
    'dark blue': '#2861ae',
    'light blue': '#add8f6',
    'purple': '#c68bdd',
    'pink': '#f9c7e2',
    'yellow': '#f6e9ad',
    'dark green': '#28ae8b'
}
colour_palette = sv.ColorPalette.from_hex(colours.values())
box_annotator = sv.BoxAnnotator(
    color=colour_palette, thickness=2, text_scale=0.6, text_thickness=1)
mask_annotator = sv.MaskAnnotator(color=colour_palette, opacity=0.65)

colours_2 = {
    'green': '#a0ff8f',
    'lime': '#c4ff94',
    'yellow': '#fcffb3',
    'grey': '#8a8a8a',
    'orange': '#edc8a1',
    'red': '#ed7b7b'
}
colour_palette_2 = sv.ColorPalette.from_hex(colours_2.values())
box_annotator_2 = sv.BoxAnnotator(
    color=colour_palette_2, thickness=2, text_scale=0.4, text_thickness=1)
mask_annotator_2 = sv.MaskAnnotator(color=colour_palette_2, opacity=0.65)

if WRITE:
    '''
    Setup SAM
    '''

    if FAST:
        sys.path.append('FastSAM')
        from fastsam import FastSAM, FastSAMPrompt
        fast_sam_model = FastSAM('FastSAM.pt')
    else:
        # sys.path.append('..')
        from segment_anything import sam_model_registry, SamPredictor

        sam_checkpoint = 'sam_vit_h_4b8939.pth'
        model_type = 'vit_h'

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=DEVICE)

        predictor = SamPredictor(sam)

if READ:
    '''
    Setup DINOv2 Image Classification
    '''

    from SegmentPlants.scripts.model import Classifier

    classification_model = Classifier(
        len(INDEX_TO_CLASS), len(INDEX_TO_SUBCLASS))
    classification_model.load_state_dict(torch.load(
        'SegmentPlants/models/subclassification_model.pt'))
    classification_model.eval()
    classification_model.to(DEVICE)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def preprocess(img):
        img = transform(img)
        img = img[None, :]
        return img


def resize_image(image: np.ndarray, expected_size: int) -> np.ndarray:
    height = image.shape[0]
    width = image.shape[1]
    new_width = expected_size
    new_height = expected_size

    if width > height:
        ratio = new_width / width
        new_height = int(height * ratio)
    else:
        ratio = new_height / height
        new_width = int(width * ratio)

    new_dimensions = (new_width, new_height)
    if ratio < 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    new_image = cv2.resize(image, new_dimensions, interpolation=interpolation)
    return new_image


def add_padding(image: Image.Image, expected_size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
    '''
    Add padding around image while repositioning a coordinate.

    :param image: PIL Image.
    :param expected_size: Expected size of new image in pixels.
    :param roi: Region of interest (coordinate) to reposition.
    :returns: A tuple of the padded image and repositioned roi.
    '''

    delta_width: int = expected_size[0] - image.size[0]
    delta_height: int = expected_size[1] - image.size[1]
    padding_width: int = delta_width // 2
    padding_height: int = delta_height // 2
    padding: tuple[int, int, int, int] = (
        padding_width, padding_height, delta_width - padding_width, delta_height - padding_height)
    new_image: Image.Image = ImageOps.expand(image, padding)
    return new_image


"""## Reading Images"""

print('\u001b[37;1m {} \u001b[37;1m'.format('Reading Images'))

image_names = []

for image_name in os.listdir(PATH):
    image_names.append(image_name)

image_names = [str(i) + '.jpg' for i in sorted([int(num.split('.')[0])
                                                for num in image_names])]

shutil.rmtree('processed', ignore_errors=True)

os.makedirs('processed', exist_ok=True)

image_paths = []

for image_name in image_names:
    image = Image.open(os.path.join(PATH, image_name))
    image = Image.fromarray(resize_image(
        np.array(ImageOps.exif_transpose(image)), 1500))
    image_path = f"processed/{image_name.split('.')[0]}.png"
    image.save(image_path)
    image_paths.append(image_path)

shutil.rmtree('outputs', ignore_errors=True)

os.makedirs('outputs', exist_ok=True)

if WRITE and WRITE_IMAGE:
    os.makedirs('outputs/grounding_dino', exist_ok=True)
    os.makedirs('outputs/sam', exist_ok=True)

if READ and READ_IMAGE:
    os.makedirs('outputs/stage', exist_ok=True)
    os.makedirs('outputs/classes', exist_ok=True)
    os.makedirs('outputs/growth', exist_ok=True)

"""# Writing"""

print('\u001b[37;1m {} \u001b[37;1m'.format('Writing'))

if WRITE:
    if CONTINUE:
        with open(CONTINUE_WRITE_PATH) as json_file:
            write_outputs = json.load(json_file)

        final_xyxy = write_outputs['xyxy']
        final_labels = write_outputs['labels']
        final_polygons = write_outputs['polygons']
    else:
        final_xyxy = []
        final_labels = []
        final_polygons = []

if WRITE:
    for count, image_path in enumerate(image_paths):
        print(f'Writing {count + 1} out of {len(image_paths)}.')

        output_name = image_path.split('/')[-1].split('.')[0]

        image_source_bgr = cv2.imread(image_path)
        image_source = cv2.cvtColor(image_source_bgr, cv2.COLOR_BGR2RGB)

        '''
        Grounding DINO
        '''

        TEXT_PROMPT = ' . '.join(CLASSES)
        TEXT_PROMPT = f'"{TEXT_PROMPT}"'

        os.system('python mmdetection/demo/image_demo.py $image_path mmdetection/configs/grounding_dino/config.py --weights SegmentPlants/models/detection_model.pth --texts $TEXT_PROMPT --device $DEVICE')

        with open(f'outputs/preds/{output_name}.json') as json_file:
            data = json.load(json_file)

        labels = np.array(data['labels'])
        scores = np.array(data['scores'])
        xyxy_grounding_dino = np.array(data['bboxes'])

        shutil.rmtree('outputs/preds')
        shutil.rmtree('outputs/vis')

        labels = np.take(np.array(CLASSES), labels).astype(str)

        boolean = scores > 0.3

        if sum(boolean) == 0:
            if WRITE_IMAGE:
                original_image = Image.fromarray(image_source)
                original_image.save(
                    f'outputs/grounding_dino/{output_name}.png')
                original_image.save(f'outputs/sam/{output_name}.png')

            final_polygons.append(None)
            final_xyxy.append(None)
            final_labels.append(None)
            continue

        scores = scores[boolean]
        labels = labels[boolean]
        xyxy_grounding_dino = xyxy_grounding_dino[boolean]

        detections_grounding_dino = sv.Detections(xyxy=xyxy_grounding_dino)

        labels_grounding_dino = [
            f'{label} {score:.2f}'
            for label, score
            in zip(labels, scores)
        ]

        if WRITE_IMAGE:
            annotated_grounding_dino = image_source_bgr.copy()
            annotated_grounding_dino = box_annotator.annotate(
                scene=annotated_grounding_dino, detections=detections_grounding_dino, labels=labels_grounding_dino)

            grounding_dino_image = Image.fromarray(
                cv2.cvtColor(annotated_grounding_dino, cv2.COLOR_BGR2RGB))
            grounding_dino_image.save(
                f'outputs/grounding_dino/{output_name}.png')

        '''
        SAM
        '''

        if FAST:
            everything_results = fast_sam_model(
                image_path, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.5)
            prompt_process = FastSAMPrompt(
                image_path, everything_results, device=DEVICE)
            ann = prompt_process.box_prompt(
                bboxes=xyxy_grounding_dino.tolist())

            class_id_sam = np.arange(len(ann))
            np.random.shuffle(class_id_sam)

            detections_sam = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=ann),
                mask=ann == 1,
                class_id=class_id_sam
            )

            labels_sam = np.array(labels_grounding_dino)
        else:
            predictor.set_image(image_source)

            transformed_boxes = predictor.transform.apply_boxes_torch(
                torch.from_numpy(xyxy_grounding_dino), image_source.shape[:2])

            centres_x = (transformed_boxes[:, 0] +
                         transformed_boxes[:, 2]) // 2
            centres_y = (transformed_boxes[:, 1] +
                         transformed_boxes[:, 3]) // 2

            centres = torch.stack((centres_x, centres_y), dim=1).to(torch.int)

            masks, iou_predictions, low_res_masks = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(DEVICE),
                multimask_output=False,
            )

            if DEVICE == 'cpu':
                detections_sam = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks=masks.numpy()[:, 0, :, :]),
                    mask=masks.numpy()[:, 0, :, :],
                    confidence=iou_predictions.numpy()[:, 0],
                    class_id=np.array(labels_grounding_dino)
                )
            else:
                detections_sam = sv.Detections(
                    xyxy=sv.mask_to_xyxy(
                        masks=masks.cpu().numpy()[:, 0, :, :]),
                    mask=masks.cpu().numpy()[:, 0, :, :],
                    confidence=iou_predictions.cpu().numpy()[:, 0],
                    class_id=np.array(labels_grounding_dino)
                )

            detections_sam = detections_sam.with_nms(
                threshold=0.5, class_agnostic=True)
            masks_sam = detections_sam.mask
            xyxy_sam = detections_sam.xyxy
            labels_sam = detections_sam.class_id
            class_id_sam = np.arange(len(labels_sam))
            np.random.shuffle(class_id_sam)

            detections_sam = sv.Detections(
                xyxy=xyxy_sam, mask=masks_sam, class_id=class_id_sam)

            predictor.reset_image()

        if WRITE_IMAGE:
            annotated_sam = mask_annotator.annotate(
                scene=image_source_bgr.copy(), detections=detections_sam)
            combined_sam = box_annotator.annotate(
                scene=annotated_sam, detections=detections_sam, labels=labels_sam)
            Image.fromarray(cv2.cvtColor(combined_sam, cv2.COLOR_BGR2RGB)).save(
                f'outputs/sam/{output_name}.png')

        # '''
        # Filter
        # '''

        masks_filter = detections_sam.mask
        xyxy_filter = detections_sam.xyxy
        labels_filter = labels_sam.copy()
        polygons_filter = []

        for mask in masks_filter:
            polygons = sv.mask_to_polygons(mask)
            for p in enumerate(polygons):
                polygons[p[0]] = p[1].tolist()
            polygons_filter.append(polygons[sorted(
                [(c, len(l)) for c, l in enumerate(polygons)], key=lambda t: t[1])[-1][0]])
        final_polygons.append(polygons_filter)

        final_xyxy.append(xyxy_filter.tolist())
        # final_masks.append(masks_filter)
        final_labels.append(labels_filter.tolist())

write_outputs = {
    'xyxy': final_xyxy,
    'polygons': final_polygons,
    'labels': final_labels
}

if (WRITE and SAVE) or (WRITE and READ):
    with open('write_outputs.json', mode='w') as f:
        json.dump(write_outputs, f)

# if WRITE and WRITE_IMAGE and SAVE:
#   !zip -r outputs.zip outputs

"""# Reading"""

print('\u001b[37;1m {} \u001b[37;1m'.format('Reading'))

if READ:
    if WRITE:
        with open('write_outputs.json') as json_file:
            write_outputs = json.load(json_file)
    else:
        with open(READ_ONLY_PATH) as json_file:
            write_outputs = json.load(json_file)

    read_xyxy = write_outputs['xyxy']
    read_polygons = write_outputs['polygons']
    read_labels = write_outputs['labels']

    if CONTINUE:
        with open(CONTINUE_READ_PATH) as json_file:
            read_outputs = json.load(json_file)
        centres = read_outputs['centres']
        areas = read_outputs['areas']
        forecasts = read_outputs['forecasts']
    else:
        centres = dict()
        areas = dict()
        forecasts = dict()

dates = pd.read_csv(DATES_PATH)

if READ:
    for i, image_path in enumerate(image_paths):
        print(f'Reading {i + 1} out of {len(image_paths)}.')

        output_name = image_path.split('/')[-1].split('.')[0]

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for key in areas:
            areas[key].append(None)
            forecasts[key].append(None)

        if read_xyxy[i] is not None and read_polygons[i] is not None:
            labels_stage = []
            class_id_stage = []

            labels_classes = []
            class_id_classes = []
            custom_colour_lookup_classes = []

            xyxy_final = []
            masks_dinov2 = []
            masks_sam = []

            labels_growth = []
            class_id_growth = []

            keys = []

            for j in range(len(read_xyxy[i])):
                masked_image = image_rgb.copy()
                bbox = read_xyxy[i][j]
                masked_image = masked_image[bbox[1]: bbox[3], bbox[0]: bbox[2]]

                '''
                Subclassification
                '''

                img = preprocess(Image.fromarray(masked_image))
                img = img.to(DEVICE)
                with torch.no_grad():
                    result = classification_model(img)
                c = result[0]
                s = result[1]
                s = s.detach()
                s = s.detach()
                c = c.cpu()
                s = s.cpu()
                c = c.numpy()
                s = s.numpy()
                c = INDEX_TO_CLASS[np.argmax(c)]
                s = INDEX_TO_SUBCLASS[np.argmax(s)]

                labels_stage.append(str((c, s)))
                if s == 'harvest' or s == 'ripe':
                    class_id_stage.append(0)
                else:
                    class_id_stage.append(3)

                xyxy_final.append(read_xyxy[i][j])
                masks_sam.append(sv.polygon_to_mask(
                    np.array(read_polygons[i][j]), (image_rgb.shape[1], image_rgb.shape[0])))

                '''
                Classes
                '''

                xyxy = read_xyxy[i][j]
                centre = (xyxy[0] + xyxy[2]) // 2, (xyxy[1] + xyxy[3]) // 2

                min_key = None
                min_dist = None
                min_centre = None
                for key in centres:
                    centre_compare = centres[key]
                    if (dist := math.sqrt(math.pow(abs(centre_compare[0] - centre[0]), 2) + math.pow(abs(centre_compare[1] - centre[1]), 2))) < DISTANCE:
                        if (min_key is not None and dist < min_dist) or (min_key is None):
                            min_key = key
                            min_dist = dist
                            min_centre = centre

                if min_key is not None:
                    centres[min_key] = min_centre
                    if min_key not in areas.keys():
                        areas[min_key] = [None] * (i + 1)
                        forecasts[min_key] = [None] * (i + 1)
                    areas[min_key][-1] = geometry.Polygon(read_polygons[i][j]).area / (
                        ARUCO_PX * ARUCO_PX) * ARUCO_CM * ARUCO_CM
                    keys.append(min_key)
                else:
                    new_key = len(centres)
                    centres[new_key] = centre
                    areas[new_key] = [None] * (i + 1)
                    forecasts[new_key] = [None] * (i + 1)
                    areas[new_key][-1] = geometry.Polygon(read_polygons[i][j]).area / (
                        ARUCO_PX * ARUCO_PX) * ARUCO_CM * ARUCO_CM
                    keys.append(new_key)

                if min_key is not None:
                    labels_classes.append(f'{min_key}, {c}')
                    class_id_classes.append(min_key)
                    custom_colour_lookup_classes.append(min_key % len(colours))
                else:
                    labels_classes.append(f'{new_key}, {c}')
                    class_id_classes.append(new_key)
                    custom_colour_lookup_classes.append(new_key % len(colours))

            '''
            Growth
            '''

            for key in keys:
                if sum(np.array(areas[key]) != None) > 2:
                    plant_areas = areas[key]

                    A_df = dates.iloc[:len(plant_areas)].copy()
                    A_df.loc[:, 'y'] = plant_areas
                    A_df['ds'] = pd.to_datetime(A_df['ds'])
                    A_df = A_df.dropna()
                    A_df['cap'] = TARGET

                    m = Prophet(growth='logistic')
                    m.fit(A_df)

                    futr_df = m.make_future_dataframe(periods=60)
                    futr_df['cap'] = TARGET * 1.1

                    forecast = m.predict(futr_df)

                    if A_df.iloc[-1, -2] >= TARGET:
                        print('\u001b[37;1m {} \u001b[37;1m'.format(
                            'Ready to harvest! 🍽'))
                        labels_growth.append('Ready to harvest!')
                        class_id_growth.append(0)
                    else:
                        if len(forecast[forecast['yhat'] >= TARGET]) > 0:
                            duration = (
                                forecast[forecast['yhat'] >= TARGET]['ds'].values[0] - A_df['ds'][0]).days
                            left = (forecast[forecast['yhat'] >= TARGET]
                                    ['ds'].values[0] - A_df['ds'][len(plant_areas) - 1]).days
                            if duration <= DAYS:
                                print('\u001b[36m {} \u001b[36m'.format(
                                    f'Growing well! ⚡'))
                                labels_growth.append(
                                    f'Growing well! {left} days till harvest.')
                                class_id_growth.append(1)
                            elif duration <= DAYS * 1.2:
                                print(
                                    '\u001b[36m {} \u001b[36m'.format(f'OK! 👌'))
                                labels_growth.append(
                                    f'OK! {left} days till harvest.')
                                class_id_growth.append(2)
                            elif duration <= DAYS * 1.4:
                                print('\u001b[35;1m {} \u001b[35;1m'.format(
                                    f'Slow growth... 🐛'))
                                labels_growth.append(
                                    f'Slow growth... {left} days till harvest.')
                                class_id_growth.append(4)
                            else:
                                print('\u001b[31m {} \u001b[31m'.format(
                                    f'Very slow growth! 🐢'))
                                labels_growth.append(
                                    f'Very slow growth! {left} days till harvest.')
                                class_id_growth.append(5)
                        else:
                            print('\u001b[31m {} \u001b[31m'.format(
                                f'Very slow growth! 🐢'))
                            labels_growth.append(
                                f'Very slow growth! {left} days till harvest.')
                            class_id_growth.append(5)
                    print('\u001b[37;1m {} \u001b[37;1m'.format(duration))

                    forecasts[key][-1] = int(duration)

                    if i == len(image_paths) - 1:
                        plot_plotly(m, forecast).show()
                else:
                    labels_growth.append('Growing...')
                    class_id_growth.append(3)

            if READ_IMAGE:
                if xyxy_final != []:
                    xyxy_final = np.array(xyxy_final)
                    masks_sam = np.array(masks_sam).astype(bool)
                    labels_stage = np.array(labels_stage)
                    class_id_stage = np.array(class_id_stage)

                    labels_growth = np.array(labels_growth)
                    class_id_growth = np.array(class_id_growth)

                    for m in ['sam']:
                        detections_stage = sv.Detections(
                            xyxy=xyxy_final,
                            mask=globals()[f'masks_{m}'],
                            class_id=class_id_stage
                        )

                        annotated_stage = mask_annotator_2.annotate(scene=image.copy(
                        ), detections=detections_stage, custom_color_lookup=class_id_stage)
                        combined_stage = box_annotator_2.annotate(
                            scene=annotated_stage, detections=detections_stage, labels=labels_stage)
                        Image.fromarray(cv2.cvtColor(combined_stage, cv2.COLOR_BGR2RGB)).save(
                            f'outputs/stage/{output_name}.png')

                        labels_classes = np.array(labels_classes)
                        class_id_classes = np.array(class_id_classes)
                        custom_colour_lookup_classes = np.array(
                            custom_colour_lookup_classes)

                        detections_classes = sv.Detections(
                            xyxy=xyxy_final,
                            mask=globals()[f'masks_{m}'],
                            class_id=class_id_classes
                        )

                        annotated_classes = mask_annotator.annotate(scene=image.copy(
                        ), detections=detections_classes, custom_color_lookup=custom_colour_lookup_classes)
                        combined_classes = box_annotator.annotate(
                            scene=annotated_classes, detections=detections_classes, labels=labels_classes)
                        Image.fromarray(cv2.cvtColor(combined_classes, cv2.COLOR_BGR2RGB)).save(
                            f'outputs/classes/{output_name}.png')

                        detections_growth = sv.Detections(
                            xyxy=xyxy_final,
                            mask=globals()[f'masks_{m}'],
                            class_id=class_id_growth
                        )

                        annotated_growth = mask_annotator_2.annotate(scene=image.copy(
                        ), detections=detections_growth, custom_color_lookup=class_id_growth)
                        combined_growth = box_annotator_2.annotate(
                            scene=annotated_growth, detections=detections_growth, labels=labels_growth)
                        Image.fromarray(cv2.cvtColor(combined_growth, cv2.COLOR_BGR2RGB)).save(
                            f'outputs/growth/{output_name}.png')
                else:
                    Image.fromarray(image_rgb).save(
                        f'outputs/stage/{output_name}.png')
                    Image.fromarray(image_rgb).save(
                        f'outputs/classes/{output_name}.png')
                    Image.fromarray(image_rgb).save(
                        f'outputs/growth/{output_name}.png')
        else:
            if READ_IMAGE:
                Image.fromarray(image_rgb).save(
                    f'outputs/stage/{output_name}.png')
                Image.fromarray(image_rgb).save(
                    f'outputs/classes/{output_name}.png')
                Image.fromarray(image_rgb).save(
                    f'outputs/growth/{output_name}.png')

read_outputs = {
    'centres': centres,
    'areas': areas,
    'forecasts': forecasts
}

with open('read_outputs.json', mode='w') as f:
    json.dump(read_outputs, f)

# if READ_IMAGE and SAVE:
#   !zip -r outputs.zip outputs

"""# Figures"""

print('\u001b[37;1m {} \u001b[37;1m'.format('Figures'))


shutil.rmtree('figures', ignore_errors=True)

os.makedirs('figures', exist_ok=True)

with open('read_outputs.json') as json_file:
    read_outputs = json.load(json_file)

for key in read_outputs['forecasts']:
    areas = read_outputs['areas'][key]
    forecasts = read_outputs['forecasts'][key]

    if sum(np.array(forecasts) != None) >= 3:
        A_df = dates.loc[:len(forecasts) - 1].copy()
        A_df['ds'] = pd.to_datetime(A_df['ds'])
        A_df.loc[:, 'y'] = forecasts
        A_df = A_df.dropna()

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=A_df.ds, y=A_df['y'], name='Forecasts', line=dict(
            color='black'), mode='lines+markers', connectgaps=True))
        fig.add_hline(y=DAYS, line_width=3,
                      line_dash="dash", line_color="#5fad95")

        fig.update_layout(title=f'Actual vs Forecasted Surface Areas',
                          xaxis_title='Date', yaxis_title='Surface Area', legend=dict(x=1, y=1))

        pio.write_html(fig, f'figures/{key}.html')

        # fig.show()

# if READ_IMAGE and SAVE:
#   !zip -r figures.zip figures