import torch

from transformers import GPT2LMHeadModel, GPT2Config, ElectraTokenizerFast

model_name = 'Kuduxaaa/gpt2-geo'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = ElectraTokenizerFast.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

prompt = 'ქართულ მითოლოგიაში '

input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
output = model.generate(
    input_ids,
    max_length = 100,
    num_beams = 5,
    no_repeat_ngram_size = 2,
    top_k = 50,
    top_p = 0.95,
    temperature = 0.7
)

result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result)
# ქართულ მითოლოგიაში, მითების პერსონაჟები და. მითები დაკავშირებული მითური წარმოშობას, რომელიც წარმოიშვა მითი გარემოც, რომ ამ პერიოდში და სხვა სხვა. აგრეთვე მითიდან წარმოადგენს მითებთან ერთად, როგორც საშუალებები, საფუძვლად წარმოების წარსულში. ლიტერატურა წარმომავლობებს მით
