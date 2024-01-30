'''
Grounding DINO
'''

import os
import torch

CLASSES : list[str] = ['mango', 'romaine lettuce', 'tomato'] # @param {type:'raw'}

TEXT_PROMPT = ' . '.join(CLASSES)
TEXT_PROMPT = f'"{TEXT_PROMPT}"'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

image_path = 'SegmentPlants/inputs/8/aligned/20.png'

os.system('python mmdetection/demo/image_demo.py $image_path mmdetection/configs/grounding_dino/config.py --weights SegmentPlants/models/detection_model.pth --texts $TEXT_PROMPT --device $DEVICE')

# import sys

# sys.path.append('FastSAM')
# from fastsam import FastSAM, FastSAMPrompt
# fast_sam_model = FastSAM('FastSAM.pt')