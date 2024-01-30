import os
import sys
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2
import torch
from torchvision import transforms
import shutil
from shapely import geometry
import json
import supervision as sv
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from huggingface_hub import hf_hub_download
from transformers import BertConfig, BertModel
from transformers import AutoTokenizer
from segment_anything import sam_model_registry, SamPredictor
from SegmentPlants.scripts.model import Classifier
import plotly.graph_objs as go
import plotly.io as pio