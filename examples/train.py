# -*- coding: utf-8 -*-

import gpt2geo

from torch.utils.data import random_split
from transformers import GPT2LMHeadModel, ElectraTokenizerFast
from datasets import load_dataset
from gpt2geo import Config

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = ElectraTokenizerFast.from_pretrained('jnz/electra-ka')
dataset = load_dataset('wikimedia/wikipedia', '20231101.ka', split='train')

text_data = dataset['text'][:Config.data_sample_size] if Config.data_sample_size is not None else dataset['text']
geo_dataset = gpt2geo.GeorgianDataset(tokenizer, text_data)

train_size = int(Config.train_size * len(geo_dataset))
val_size = len(geo_dataset) - train_size
train_dataset, val_dataset = random_split(geo_dataset, [train_size, val_size])

trainer = gpt2geo.GPT2GeoLMHead(model, tokenizer)
trainer.train(train_dataset, val_dataset, epochs=20)

generated_text = trainer.inference(
    prompt = 'ქართულ მითოლოგიაში ',
    max_length = 100,
    num_beams = 5,
    no_repeat_ngram_size = 2,
    top_k = 50,
    top_p = 0.95,
    temperature = 0.7
)

print(generated_text)
# ქართულ მითოლოგიაში, მითების პერსონაჟები და. მითები დაკავშირებული მითური წარმოშობას, რომელიც წარმოიშვა მითი გარემოც, რომ ამ პერიოდში და სხვა სხვა. აგრეთვე მითიდან წარმოადგენს მითებთან ერთად, როგორც საშუალებები, საფუძვლად წარმოების წარსულში. ლიტერატურა წარმომავლობებს მით