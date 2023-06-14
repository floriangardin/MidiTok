from miditok.config import Config
from miditok.funcs import create_dataset

if __name__ == "__main__":
    config: Config = Config("config.json")

    # Load all arguments and update the config

    dataset_path = config.dataset_path
    tokens_path = config.tokens_path
    tokens_bpe_path = config.tokens_bpe_path
    tokenizer_path = config.tokenizer_path

    create_dataset(dataset_path, tokens_path, tokens_bpe_path, tokenizer_path, finetune=False)
