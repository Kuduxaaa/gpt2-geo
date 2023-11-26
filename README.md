# GPT2GeoLMHead
`🇬🇪 <GPT2GeoLMHead>`

### Overview:

The GPT2GeoLMHead project is a natural language processing (NLP) venture aimed at training a Georgian language model head utilizing the powerful GPT-2 architecture. This initiative involves the utilization of state-of-the-art transformer-based language models to understand and generate coherent and contextually relevant Georgian text.

### Components:

#### 1. **Data Collection and Preparation:**
   - The project leverages the [Wikimedia/Wikipedia](https://huggingface.co/datasets/wikimedia) dataset for training the language model.
   - The dataset is loaded using the Hugging Face `datasets` library, focusing on the Georgian language (`ka`) split from November 1, 2023.

#### 2. **Dataset Processing:**
   - A custom dataset, `GeorgianDataset`, is created to preprocess and tokenize the Georgian text data using the ElectraTokenizerFast.
   - The dataset supports dynamic block sizing and includes the necessary special tokens for proper model training.

#### 3. **Model Configuration:**
   - The GPT-2 model, pretrained on a diverse range of internet text, is utilized as the base architecture.
   - A specific instance of the ElectraTokenizerFast is employed for tokenization in alignment with the model's requirements.
   - Special tokens, including `<pad>`, `<eos>`, and `<mask>`, are added to enhance model understanding and context.

#### 4. **Training Loop:**
   - The training process is orchestrated by the `GPT2GeoLMHead` class, which encapsulates the GPT-2 model, tokenizer, and training logic.
   - Training hyperparameters, such as learning rate, batch size, and the number of epochs, are configured in a dedicated `Config` class.
   - The training loop involves both training and validation phases, utilizing PyTorch's DataLoader for efficient data handling.
   - CrossEntropyLoss is employed as the loss function, and the AdamW optimizer is used for gradient descent.

#### 5. **Inference:**
   - The model includes an inference method, allowing users to generate Georgian text given a prompt.
   - Various parameters, such as `num_beams` and `temperature`, can be adjusted to influence the diversity and creativity of generated text.

#### 6. **Model Saving:**
   - The project incorporates functionality to save the pretrained GPT-2 model and tokenizer for future use.

![#loss](https://raw.githubusercontent.com/Kuduxaaa/gpt2-geo/main/loss-stat.png?token=GHSAT0AAAAAACJGA7TBE34256UQI6CBNKTMZLDPWMQ)

### Usage:

1. **Initialization:**
   - Create an instance of `GPT2GeoLMHead` by providing the GPT-2 model and ElectraTokenizerFast.
   - Configure training parameters and dataset size in the `Config` class.

2. **Training:**
   - Call the `train` method on the `GPT2GeoLMHead` instance, passing the training and validation datasets.

3. **Inference:**
   - Utilize the `inference` method to generate Georgian text based on a given prompt.

4. **Model Saving:**
   - Save the pretrained model and tokenizer using the `save_pretrained` method.

### Conclusion:

The GPT2GeoLMHead project showcases the application of cutting-edge language models for understanding and generating Georgian text. Its modular and object-oriented design facilitates easy extension and integration into various NLP applications. The project stands as a testament to the synergy of transformer architectures and the rich linguistic diversity encapsulated by the Georgian language.
