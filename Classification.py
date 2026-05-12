import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet169
import numpy as np

class PatchExtractor(layers.Layer):
    def __init__(self, patch_size):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images,
                                           sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, self.patch_size, self.patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches



class PositionalEncoding(layers.Layer):
    def __init__(self, max_patches, projection_dim):
        super(PositionalEncoding, self).__init__()
        self.projection_dim = projection_dim
        self.position_embedding = layers.Embedding(input_dim=max_patches, output_dim=self.projection_dim)

    def call(self, patches):
        # Dynamically compute the number of patches from the input patches
        num_patches = tf.shape(patches)[1]

        # Generate positions for each patch
        positions = tf.range(start=0, limit=num_patches, delta=1)

        # Encode the positions and add them to the patch embeddings
        encoded_positions = self.position_embedding(positions)
        return patches + encoded_positions




# 4. Transformer Encoder
def transformer_encoder(patches, projection_dim, num_heads):
    x = layers.LayerNormalization(epsilon=1e-6)(patches)
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x, x)
    x = layers.Add()([attention_output, patches])
    
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    mlp_output = layers.Dense(units=projection_dim, activation=tf.nn.gelu)(x)
    return mlp_output

# 5. Decoder Block
def decoder_block(input_tensor, encoder_output, projection_dim, num_heads):
    x = layers.LayerNormalization(epsilon=1e-6)(input_tensor)
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x, encoder_output)
    x = layers.Add()([attention_output, input_tensor])
    
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    mlp_output = layers.Dense(units=projection_dim, activation=tf.nn.gelu)(x)
    return mlp_output

# 6. Final Model
def DCEPVTM(x_train,y_train):
    image_size = x_train[0].shape[0]
    patch_size = 16
    projection_dim = 64
    num_heads = 4
    num_encoder_layers = 4
    num_decoder_layers = 4
    input_shape=x_train[0].shape
    base_model = DenseNet169(include_top=False, input_shape=(256,256,3))
    
    inputs = layers.Input(shape=(image_size, image_size, 3))
    
     # DenseNet feature extraction without pooling
    features = base_model(inputs)
     
    # Compute the number of patches based on feature map dimensions
    feature_map_size = features.shape[1]
    num_patches = (feature_map_size // patch_size) ** 2
    
    # Patch extraction
    patches = PatchExtractor(patch_size=patch_size)(features)
    
    # Encode the patches with a dense layer
    encoded_patches = layers.Dense(units=projection_dim)(patches)
    
    max_patches = 64  # Adjust this based on your expected number of patches
    projection_dim = 64  # Set the desired projection dimension
    positional_encoding_layer = PositionalEncoding(max_patches=max_patches, projection_dim=projection_dim)

    # Encoder
    for _ in range(num_encoder_layers):
        encoded_patches = transformer_encoder(encoded_patches, projection_dim, num_heads)
    
    # Decoder
    x = encoded_patches
    for _ in range(num_decoder_layers):
        x = decoder_block(x, encoded_patches, projection_dim, num_heads)
    
    # Reshape or Pool the decoder output
    x = layers.GlobalAveragePooling1D()(x)
    
   
    output = layers.Dense(units=len(np.unique(y_train))-1, activation="sigmoid")(x)
      
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
   
    return model
