import pandas as pd
import numpy as np
from tensorflow import keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K

# 读取原始的.csv文件
df = pd.read_csv("..\\datasets\\normalized_data.csv")

# 获取特征列的数据
features = df.iloc[:, 1:].values

# 定义输入层
input_shape = features.shape[1]
input_layer = Input(shape=(input_shape,))
# 定义编码器
latent_dim = 600  # 定义潜在变量的维度
hidden_layer_1 = Dense(256, activation='relu')(input_layer)
hidden_layer_2 = Dense(128, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(64, activation='relu')(hidden_layer_2)
z_mean = Dense(latent_dim)(hidden_layer_3)
z_log_var = Dense(latent_dim)(hidden_layer_3)
# 定义潜在变量采样函数
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon
# 定义解码器
decoder_hidden_1 = Dense(64, activation='relu')
decoder_hidden_2 = Dense(128, activation='relu')
decoder_hidden_3 = Dense(256, activation='relu')
decoder_output = Dense(input_shape)
# 定义潜在空间采样层
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# 连接编码器和解码器
decoder_input = decoder_hidden_1(z)
decoder_input = decoder_hidden_2(decoder_input)
decoder_input = decoder_hidden_3(decoder_input)
decoder_output = decoder_output(decoder_input)
# 定义VAE模型
vae = Model(input_layer, decoder_output)
# 定义重构误差和KL散度
reconstruction_loss = keras.losses.mse(input_layer, decoder_output)
reconstruction_loss *= input_shape
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# 训练VAE模型
vae.fit(features, epochs=150, batch_size=32)

# 获取特征提取器
feature_extractor = Model(input_layer, z_mean)

# 对特征列进行特征提取
encoded_features = feature_extractor.predict(features)

# 将提取的特征写入新的.csv文件中
feature_columns = ["feature_{}".format(i) for i in range(encoded_features.shape[1])]
df_new = pd.DataFrame(encoded_features, columns=feature_columns)
df_new.insert(loc=0, column='entrezId', value=df.iloc[:, 0])
df_new.to_csv("..\\datasets\\features_epoch=150_ycc=3.csv", index=False)