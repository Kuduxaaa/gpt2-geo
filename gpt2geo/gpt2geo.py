import os
import torch

from .config import Config
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class GPT2GeoLMHead:
    def __init__(self, model, tokenizer):
        self.device = Config.device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        special_tokens = {
            'pad_token': '<pad>',
            'eos_token': '<eos>',
            'mask_token': '<mask>'
        }

        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))


    def __repr__(self):
        return '🇬🇪 <GPT2GeoLMHead>'


    def collate_batch(self, batch):
        return pad_sequence(
            batch,
            batch_first = True,
            padding_value = self.tokenizer.pad_token_id
        )


    def train(self,
              train_dataset,
              val_dataset,
              epochs = 5,
              batch_size = Config.batch_size,
              learning_rate = Config.lr):

        train_dataloader = DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            collate_fn = self.collate_batch
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size = batch_size,
            collate_fn = self.collate_batch
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        print('✨ Starting training...')

        for epoch in range(epochs):
            self.train_one_epoch(train_dataloader, optimizer, criterion)
            avg_val_loss = self.validate(val_dataloader)

            print(f'🍀 Epoch {epoch + 1}/{epochs}, Avg Validation Loss: {avg_val_loss}')

        print('🚀 Training complete! ')


    def train_one_epoch(self, dataloader, optimizer, criterion):
        self.model.train()

        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch.to(self.device)
            labels = input_ids.clone()
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()


    def validate(self, dataloader):
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch.to(self.device)
                labels = input_ids.clone()
                outputs = self.model(input_ids, labels=labels)
                total_val_loss += outputs.loss.item()

        return total_val_loss / len(dataloader)


    def inference(self,
                  prompt,
                  max_length=100,
                  num_beams=5,
                  no_repeat_ngram_size=2,
                  top_k=50,
                  top_p=0.95,
                  temperature=0.7):

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)


    def model(self):
        return self.model


    def tokenizer(self):
        return self.tokenizer


    def save_pretrained(self, directory):
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
