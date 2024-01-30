# Description
Monitoring crops using [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), [SAM](https://github.com/facebookresearch/segment-anything), and [DINOv2](https://github.com/facebookresearch/dinov2).

# Other References:
 - [mmdetection](https://github.com/open-mmlab/mmdetection)
 - [Downstream DINOv2](https://github.com/itsprakhar/Downstream-Dinov2)

# Training:

Modify data folder for custom dataset. Finetuned models can be found [here](https://huggingface.co/GreeneryScenery/Chronocrop).

<details>
<summary> <strong> Finetune Grounding DINO </strong> </summary>
<br/>

1. Clone mmdetection.
``` console
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/multimodal.txt
pip install -U openmim
mim install mmengine mmdet mmcv
```

2. Clone this repository.
``` console
git clone https://github.com/GreeneryScenery/SegmentPlants.git
```
3. Download BERT.
``` python
from transformers import BertConfig, BertModel
from transformers import AutoTokenizer

config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, config=config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

config.save_pretrained("bert-base-uncased")
model.save_pretrained("bert-base-uncased")
tokenizer.save_pretrained("bert-base-uncased")
```

4. Move config.py.
``` python
import shutil

shutil.move("SegmentPlants/inputs/data/detection/config/config.py", "mmdetection/configs/grounding_dino/config.py")
``` 

5. Run training script.
``` console
bash ./tools/dist_train.sh configs/grounding_dino/config.py 1 --work-dir detection_work_dir
```

</details>

<details>
<summary> <strong> Downstream DINOv2 </strong> </summary>
<br/>

1. Clone this repository.
``` console
git clone https://github.com/GreeneryScenery/SegmentPlants.git
cd SegmentPlants
```

2. Run training scripts.
``` console
python scripts/train_classifier.py #Classification
python scripts/train_segmentor.py #Segmentation
python scripts/train_localiser.py #Localisation
```

</details>

# Usage:
Example notebook [here](scripts/Example.ipynb)!
