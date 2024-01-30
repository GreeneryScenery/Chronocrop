from huggingface_hub import hf_hub_download

hf_hub_download(repo_id = 'GreeneryScenery/Chronocrop', filename = 'detection_model.pth', local_dir = 'SegmentPlants/models')
hf_hub_download(repo_id = 'GreeneryScenery/Chronocrop', filename = 'subclassification_model.pt', local_dir = 'SegmentPlants/models')

# '''
# Setup Grounding DINO
# '''

# from transformers import BertConfig, BertModel
# from transformers import AutoTokenizer

# config = BertConfig.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased', add_pooling_layer = False, config = config)
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# config.save_pretrained('mmdetection/bert-base-uncased')
# model.save_pretrained('mmdetection/bert-base-uncased')
# tokenizer.save_pretrained('mmdetection/bert-base-uncased')