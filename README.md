## Caption Generation Model

### Pre-Trained CNN Architecture for Transformer-Based Image Caption Generation Model

This project leverages a Keras/TensorFlow-based image captioning application using CNN (Convolutional Neural Networks) for feature extraction and Transformer networks (Encoder-Decoder) for language modeling. In particular, the architecture consists of three key models:
1. **CNN (EfficientNetB0)**: The CNN is responsible for extracting features from images. We use **EfficientNetB0**, which is pre-trained on ImageNet, for robust and efficient image feature extraction.
2. **TransformerEncoder**: The extracted image features are passed to a Transformer encoder. This encoder processes the image features and generates a contextual representation that captures important information from the image.
3. **TransformerDecoder**: The decoder takes both the encoder's output (the image features) and the textual data (captions) as inputs. It tries to predict the caption for the image, learning to generate grammatically correct and semantically accurate descriptions.

### Notes:
- You can run this code on any Python platform, such as **PyCharm** or any online platform that supports TensorFlow/Keras.

### Requirements:
The following versions of the libraries are required for smooth execution of the code:
- Python: >= 3.6
- Libraries:
  - numpy
  - seaborn
  - keras
  - tensorflow
  - tqdm
  - nltk

### Dataset

The **Flickr30k dataset** is a widely used benchmark dataset for image captioning tasks in the field of computer vision and natural language processing (NLP). It consists of 31,783 images, each paired with five English descriptions. The dataset provides a diverse range of images from various categories and is used to evaluate image captioning models where the goal is to generate descriptive captions for the images.

### Architecture
Our proposed architecture follows the **Transformer-based model** for image captioning, where we combine a CNN for feature extraction and a Transformer network for language modeling.

#### Main Components of the Transformer Architecture:
- **Encoder**: Processes the image features and generates a contextual representation.
- **Decoder**: Takes the encoder output along with the textual sequence and generates the caption.
- **Positional Encoding**: Since Transformers do not have any inherent sense of order, positional encoding is added to the input sequences to preserve the order information.
- **Embeddings**: Both image features and text tokens are embedded to be input to the Transformer.
- **Multi-Headed Attention**: This mechanism allows the model to focus on different parts of the input sequence at each step, enhancing its capacity to learn from different perspectives.

<p align="center">
  <img src="https://github.com/user-attachments/assets/76c799fc-7281-40c2-8135-40e6b255ab27" />
</p>
*Figure 1: Overview of the Transformer-based Image Captioning Model. The image features are extracted via EfficientNetB0 (CNN) and passed to the Transformer Encoder, which then feeds the Transformer Decoder to generate captions.*

<p align="center">
  <img src="https://github.com/user-attachments/assets/b18076c5-1601-4bf8-85ec-c9d092575685" />
</p>
*Figure 2: Transformer Encoder-Decoder Architecture. This diagram highlights the flow of data from image feature extraction to caption generation. The image features are transformed into a sequence of vectors by the encoder and used by the decoder to predict the next words in the caption.*

### Results

The model's performance is evaluated using the **BLEU score** (Bilingual Evaluation Understudy), which is a standard metric for evaluating the quality of machine-generated text, particularly for machine translation and caption generation tasks. For this project, we used **greedy decoding**, but it’s noted that better results can be achieved by implementing **beam search** for more advanced decoding strategies.

#### Caption Examples:
The **qualitative outcomes** of the model's inference show that the Transformer-based model generates grammatically valid captions for images. The images may not have been present in the prior training dataset, but the model can generate appropriate and sensible captions based on its learned features. Although the BLEU scores are not perfect, they provide a reasonable reference for evaluating the model's ability to describe unseen images.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e1826cfe-2eba-4537-b166-238ac7042b87" />
</p>
*Figure 3: Example of generated captions by the Transformer-based model. These captions are generated for images that were not present in the training dataset, demonstrating the model's ability to generalize.*

The **validation accuracy** of the model increases with each epoch, indicating that the model is learning effectively during training.

<p align="center">
  <img src="https://github.com/user-attachments/assets/86d56980-5945-4b45-b4ec-30ff3e714efb" />
</p>
*Figure 4: Training and validation accuracy over epochs. As the model trains, we see an increase in the accuracy on both the training and validation sets, which suggests that the model is learning and generalizing well.*


