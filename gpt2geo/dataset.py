import re
import string

from torch.utils.data import Dataset

class GeorgianDataset(Dataset):
    def __init__(self, 
                 tokenizer, 
                 data, 
                 block_size = Config.dataset_block_size):
      
        self.data = self.prepare_dataset(tokenizer, data, block_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.tensor(self.data[item], dtype=torch.long)

    def preprocess_text(self, input_text):
        input_text = input_text.lower()
        symbols_to_remove = re.escape(string.punctuation)
        input_text = re.sub(f'[{symbols_to_remove}]', '', input_text)
        input_text = ' '.join(input_text.split())

        return input_text

    def prepare_dataset(self, tokenizer, data, block_size):
        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        return [tokenizer.encode(f'{self.preprocess_text(text)}{tokenizer.eos_token}', max_length=block_size, truncation=True) for text in data]
            tokenizer.encode(
                f'{text}{tokenizer.eos_token}', 
                max_length = block_size, 
                truncation = True
            ) for text in data
        ]
