#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import os


# In[2]:


BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH = 100


# In[3]:


def load_data_file():
    en_file = [
        './wmt19_translate/zh-en/UNv1.0.en-zh.en',
        './wmt19_translate/zh-en/train.en'
    ]
    zh_file = [
        './wmt19_translate/zh-en/UNv1.0.en-zh.zh',
        './wmt19_translate/zh-en/train.zh'
    ]
    example_en = tf.data.TextLineDataset(en_file, num_parallel_reads=3)
    example_zh = tf.data.TextLineDataset(zh_file, num_parallel_reads=3)
    examples = tf.data.Dataset.zip((example_en, example_zh))
    return examples


# In[4]:


def train_val(examples,train_size = 128000,
        val_size = 25600):
    train_examples = examples.take(train_size)
    val_examples = examples.skip(train_size).take(val_size)
    return train_examples,val_examples


# In[5]:


def create_tokenizer(train_examples):
    import time
    start = time.time()
    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for en, zh in train_examples), target_vocab_size=2**16)
    tokenizer_en.save_to_file('./vocab/enVOCAB')
    log('create eng!', time.time() - start)
    start = time.time()
    tokenizer_zh = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (zh.numpy() for en, zh in train_examples), target_vocab_size=2**16)
    tokenizer_zh.save_to_file('./vocab/zhVOCAB')
    log('create zh!', time.time() - start)
    return tokenizer_en, tokenizer_zh


def create_en(train_examples):
    start = time.time()
    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for en, zh in train_examples), target_vocab_size=2**16)
    tokenizer_en.save_to_file('./vocab/enVOCAB')
    log('create eng!', time.time() - start)
    return tokenizer_en


def create_zh(train_examples):
    start = time.time()
    print('start')
    tokenizer_zh = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (zh.numpy() for en, zh in train_examples),
        target_vocab_size=2**16,
        max_subword_length=5)
    tokenizer_zh.save_to_file('./vocab/zhVOCAB')
    log('create zh!', time.time() - start)
    return tokenizer_zh


# In[6]:


def load_tokenizer(path):
    tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(
        path + '/enVOCAB')
    tokenizer_zh = tfds.features.text.SubwordTextEncoder.load_from_file(
        path + '/zhVOCAB')
    return tokenizer_en, tokenizer_zh


# In[7]:


def encode(lang1, lang2):
    lang1 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang1.numpy()) + [tokenizer_en.vocab_size + 1]

    lang2 = [tokenizer_zh.vocab_size] + tokenizer_zh.encode(
        lang2.numpy()) + [tokenizer_zh.vocab_size + 1]

    return lang1, lang2


# In[8]:


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


# In[10]:


def tf_encode(en, zh):
    result_en, result_zh = tf.py_function(encode, [en, zh],
                                          [tf.int64, tf.int64])
    result_en.set_shape([None])
    result_zh.set_shape([None])

    return result_en, result_zh


# In[9]:


def creat_datasets(train_examples, val_examples,train_size):
    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    train_dataset = train_dataset.take(train_size)
    # 将数据集缓存到内存中以加快读取速度。
    train_dataset = train_dataset.cache('cache/cache0')
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
        BATCH_SIZE, padded_shapes=(MAX_LENGTH, MAX_LENGTH))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(
        BATCH_SIZE, padded_shapes=(MAX_LENGTH, MAX_LENGTH))
    return train_dataset, val_dataset


# In[11]:


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


# In[14]:


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :], d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# In[13]:


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# In[12]:


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# In[15]:


def scaled_dot_product_attention(q, k, v, mask):
    """计算注意力权重。
  q, k, v 必须具有匹配的前置维度。
  k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
  虽然 mask 根据其类型（填充或前瞻）有不同的形状，
  但是 mask 必须能进行广播转换以便求和。
  
  参数:
    q: 请求的形状 == (..., seq_len_q, depth)
    k: 主键的形状 == (..., seq_len_k, depth)
    v: 数值的形状 == (..., seq_len_v, depth_v)
    mask: Float 张量，其形状能转换成
          (..., seq_len_q, seq_len_k)。默认为None。
    
  返回值:
    输出，注意力权重
  """

    matmul_qk = tf.matmul(q, k,
                          transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
    # 相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits,
                                      axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# In[17]:


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
    转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
    """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask,cache = None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        if cache is not None:
            if cache.get('k') == None:
                cache['k'] = k
                cache['v'] = v
            else:
#                 k = tf.concat([cache['k'], k], axis=-1)
#                 v = tf.concat([cache['v'], v], axis=-1)
            
                cache['k'] = k
                cache['v'] = v

        q = self.split_heads(
            q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(
            k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(
            v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(
            scaled_attention,
            perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(
            concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


# In[16]:


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff,
                              activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# In[20]:


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x,
                                  mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(
            x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


# In[19]:


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask,cache = None):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(
            x, x, x, look_ahead_mask,cache = cache)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1,
            padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 +
                               out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output +
                               out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


# In[18]:


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 maximum_position_encoding,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


# In[22]:


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 target_vocab_size,
                 maximum_position_encoding,
                 rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask,cache = None):

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            if cache is not None:
                layer_name = 'layers_%s'%i
                cache[layer_name] = {}
                layer_cache = cache[layer_name] 
            else:
                layer_cache = None
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask,
                                                   padding_mask,
                                                  cache = layer_cache)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


# In[21]:


class Transformer(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 target_vocab_size,
                 pe_input,
                 pe_target,
                 rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask,
             dec_padding_mask):

        enc_output = self.encoder(
            inp, training,
            enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training,
                                                     look_ahead_mask,
                                                     dec_padding_mask)

        final_output = self.final_layer(
            dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
    
    def predict(self,x):
        start_token = [tokenizer_en_vocab_size]
        end_token = [tokenizer_en_vocab_size + 1]
        
        # 输入语句是葡萄牙语，
        #增加开始和结束标记
        x = start_token + tokenizer_en.encode(x) + end_token
        encoder_input = tf.expand_dims(x, 0)

        # 因为目标是英语，输入 transformer 的第一个词应该是
        # 英语的开始标记。
        decoder_input = [tokenizer_zh_vocab_size]
        output = tf.expand_dims(decoder_input, 0)
        
        enc_output = None
        predicted_id = output
        global cache
        cache = {}
        for i in range(MAX_LENGTH):
            if i==0:
                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            look_ahead_mask = create_look_ahead_mask(tf.shape(output)[1])
            dec_target_padding_mask = create_padding_mask(output)
            combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
                
            if enc_output == None:
                enc_output = self.encoder(
                encoder_input, False,
                enc_padding_mask)
                
            dec_output, attention_weights = self.decoder(output,
                                                     enc_output,
                                                     False,
                                                     combined_mask,
                                                     dec_padding_mask,
                                                    cache = None)

            predictions = self.final_layer(
                dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
#             print(cache)
            # 从 seq_len 维度选择最后一个词
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    #         print(cache)
            # 如果 predicted_id 等于结束标记，就返回结果
            if predicted_id == tokenizer_zh.vocab_size + 1:
                break
            # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
            output = tf.concat([output, predicted_id], axis=-1)
            result = output
            
            
        result = tf.squeeze(result, axis=0)
        predicted_sentence = tokenizer_zh.decode(
        [i for i in result if i < tokenizer_zh.vocab_size])
        
        return predicted_sentence
    
    def fit(self):
        pass


# In[23]:


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# In[27]:


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# In[24]:


def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)

    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp)

    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# In[25]:


def create_ckpt(checkpoint_path = "./checkpoints/13_100_light"):
    

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        log('Latest checkpoint restored!!')
    return ckpt,ckpt_manager


# In[26]:


# 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地
# 执行。该函数专用于参数张量的精确形状。为了避免由于可变序列长度或可变
# 批次大小（最后一批次较小）导致的再追踪，使用 input_signature 指定
# 更多的通用形状。

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask,
                                     combined_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


# In[28]:


from datetime import datetime
#日志保存函数
def log(*args,fn:str=''):
    sentence = ''
    if fn=='':
        fn='./log.txt'
    for a in args:
        sentence+=str(a)
        sentence+=' '
    nowTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(fn, 'a') as f:
        f.writelines(sentence + '\t%s\n' % nowTime)
    print(sentence)


# In[29]:


class Translator():
    def __init__(self,train_size = 0,val_size = 0,ckpt_path = 'checkpoints/13_100_light',
                vocab_path = 'vocab/13_100_light'):
        global train_examples, val_examples
        global train_dataset, val_dataset
        global tokenizer_en, tokenizer_zh
        global ckpt, ckpt_manager
        examples = load_data_file()
        train_examples,val_examples = train_val(examples,train_size,val_size)
        train_dataset, val_dataset = creat_datasets(train_examples=examples,
                                                    val_examples=val_examples,
                                                 train_size=train_size)

        tokenizer_en, tokenizer_zh = load_tokenizer(vocab_path)

        # 超参数（hyperparameters） 为了让本示例小且相对较快，已经减小了num_layers、 d_model 和 dff 的值。
        # Transformer 的基础模型使用的数值为：
        #    num_layers=6，d_model = 512，dff = 2048。
        #     num_layers = 6
        #     d_model = 512
        #     dff = 2048
        #     num_heads = 8

        num_layers = 4
        d_model = 128
        dff = 512
        num_heads = 8
        
        global tokenizer_en_vocab_size,tokenizer_zh_vocab_size
    
        tokenizer_en_vocab_size = tokenizer_en.vocab_size
        tokenizer_zh_vocab_size = tokenizer_zh.vocab_size
        input_vocab_size = tokenizer_en_vocab_size + 2
        target_vocab_size = tokenizer_zh_vocab_size + 2

        self.transformer = Transformer(num_layers,
                                  d_model,
                                  num_heads,
                                  dff,
                                  input_vocab_size,
                                  target_vocab_size,
                                  pe_input=input_vocab_size,
                                  pe_target=target_vocab_size,
                                      )
        ckpt = tf.train.Checkpoint(transformer=self.transformer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=5)

        # 如果检查点存在，则恢复最新的检查点。
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            log('Latest checkpoint restored!!')

        
    def translate(self,inp_sentence):
        return self.transformer.predict(inp_sentence)


# In[31]:


# t = None
# t = Translator()


# In[30]:


# %timeit t.translate("""3. The existence of support services for women who are the victims of aggression or abuses;3. The existence of support services for women who are the victims of aggression or abuses;3. The existence of support services for women who are the victims of aggression or abuses;""")


# In[33]:


# %timeit t.translate('i will love you forever!')


# In[ ]:




