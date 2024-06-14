from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import io
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the tokenizer
from keras.layers import TextVectorization
from_disk = pickle.load(open('C://test//eye_speak1//Backend//tv_layer.pkl', "rb"))
tokenizer = TextVectorization.from_config(from_disk['config'])
tokenizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
tokenizer.set_weights(from_disk['weights'])

word2idx = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())
idx2word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)

# Define model architecture
MAX_LENGTH = 40
VOCABULARY_SIZE = 15000
EMBEDDING_DIM = 512
UNITS = 512

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")

    def call(self, x, training):
        x = self.layer_norm_1(x)
        x = self.dense(x)
        attn_output = self.attention(query=x, value=x, key=x, attention_mask=None, training=training)
        x = self.layer_norm_2(x + attn_output)
        return x

class Embeddings(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(max_len, embed_dim)

    def call(self, input_ids):
        length = tf.shape(input_ids)[-1]
        position_ids = tf.range(start=0, limit=length, delta=1)
        position_ids = tf.expand_dims(position_ids, axis=0)
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        return token_embeddings + position_embeddings

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, units, num_heads):
        super().__init__()
        self.embedding = Embeddings(tokenizer.vocabulary_size(), embed_dim, MAX_LENGTH)
        self.attention_1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.attention_2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()
        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)
        self.out = tf.keras.layers.Dense(tokenizer.vocabulary_size(), activation="softmax")
        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.5)

    def call(self, input_ids, encoder_output, training, mask=None):
        embeddings = self.embedding(input_ids)
        combined_mask = None
        padding_mask = None
        if mask is not None:
            causal_mask = self.get_causal_attention_mask(embeddings)
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        attn_output_1 = self.attention_1(query=embeddings, value=embeddings, key=embeddings, attention_mask=combined_mask, training=training)
        out_1 = self.layernorm_1(embeddings + attn_output_1)
        attn_output_2 = self.attention_2(query=out_1, value=encoder_output, key=encoder_output, attention_mask=padding_mask, training=training)
        out_2 = self.layernorm_2(out_1 + attn_output_2)
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, cnn_model, encoder, decoder):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        image_inputs, caption_inputs = inputs
        image_features = self.cnn_model(image_inputs)
        encoder_output = self.encoder(image_features)
        decoder_output = self.decoder(caption_inputs, encoder_output)
        return decoder_output

def CNN_Encoder():
    inception_v3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    output = inception_v3.output
    output = tf.keras.layers.Reshape((-1, output.shape[-1]))(output)
    cnn_model = tf.keras.models.Model(inception_v3.input, output)
    return cnn_model

encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)
decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)
cnn_model = CNN_Encoder()
caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)

logging.info("Loading model weights...")
caption_inputs = np.random.rand(1, 40)
image_inputs = np.random.rand(1, 299, 299, 3)
_ = caption_model([image_inputs, caption_inputs])
caption_model.load_weights('C://test//eye_speak1//Backend//Image_Captioning_Weights_f.h5')
logging.info("Model weights loaded successfully.")

def load_image_from_path(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

def generate_caption(img, add_noise=False):
    logging.info("Generating caption for the image.")
    if add_noise:
        noise = tf.random.normal(img.shape) * 0.1
        img = img + noise
        img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
    img = tf.expand_dims(img, axis=0)
    img_embed = caption_model.cnn_model(img)
    img_encoded = caption_model.encoder(img_embed, training=False)
    y_inp = '[start]'
    for i in range(MAX_LENGTH - 1):
        tokenized = tokenizer([y_inp])[:, :-1]
        mask = tf.cast(tokenized != 0, tf.int32)
        pred = caption_model.decoder(tokenized, img_encoded, training=False, mask=mask)
        pred_idx = np.argmax(pred[0, i, :])
        pred_idx = tf.convert_to_tensor(pred_idx)
        pred_word = idx2word(pred_idx).numpy().decode('utf-8')
        if pred_word == '[end]':
            break
        y_inp += ' ' + pred_word
    y_inp = y_inp.replace('[start] ', '')
    logging.info("Generated caption: %s", y_inp)
    return y_inp

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((299, 299))
    img = np.array(img)
    if img.shape[2] == 4:  # Handle PNG with transparency
        img = img[..., :3]
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    caption = generate_caption(img)
    return jsonify({'caption': caption})

@app.route("/test")
def test_connection():
    return "Connection successful!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
