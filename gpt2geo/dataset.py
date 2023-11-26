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
        return torch.tensor(
            self.data[item], 
            dtype = torch.long
        )

    def prepare_dataset(self, tokenizer, data, block_size):
        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        return [
            tokenizer.encode(
                f'{text}{tokenizer.eos_token}', 
                max_length = block_size, 
                truncation = True
            ) for text in data
        ]