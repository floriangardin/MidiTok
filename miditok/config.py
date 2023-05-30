import json
import os

"""
Config class to load the config file
Config example : 

{
  "base_model_path": "/Users/floriangardin/code/music/musiclang2/locals/data",
  "dataset_path": "data_composers",
  "tokens_path": "tokens",
  "tokens_bpe_path": "tokens_bpe",
  "tokens_split_path": "data",
  "model_path": "model",
  "tokenizer_path": "tokenizer.json"
}

"""
class Config:

    def __init__(self, config_file):

        with open(config_file, 'r') as f:
            config = json.load(f)

        self.config = config

        os.makedirs(self.base_model_path, exist_ok=True)
        os.makedirs(self.tokens_path, exist_ok=True)
        os.makedirs(self.tokens_bpe_path, exist_ok=True)
        os.makedirs(self.tokens_split_path, exist_ok=True)

    @property
    def base_model_path(self):
        return self.config['base_model_path']

    @property
    def dataset_path(self):
        return self.config['dataset_path']

    @property
    def tokens_path(self):
        return os.path.join(self.base_model_path, self.config['tokens_path'])

    @property
    def tokens_bpe_path(self):
        return os.path.join(self.base_model_path, self.config['tokens_bpe_path'])

    @property
    def tokens_split_path(self):
        return os.path.join(self.base_model_path, self.config['tokens_split_path'])

    @property
    def model_path(self):
        return os.path.join(self.base_model_path, self.config['model_path'])

    @property
    def tokenizer_path(self):
        return os.path.join(self.base_model_path, self.config['tokenizer_path'])


