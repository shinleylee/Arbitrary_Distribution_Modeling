import math
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Flatten, Concatenate, Softmax
from tensorflow.keras.layers import Input, Dense, Lambda, Embedding
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import horovod.tensorflow.keras as hvd
from deepctr.layers.interaction import InteractingLayer


embedding_paras = [
    # embedding column name, distinct value count, embedding dimensionality
    # request
    ("server_id", 800, 9),
    ("video_group_id", 20, 4),
    ("site_group_id", 30, 5),
    ("distributor_group_id", 20, 4),
    ("time_position_class", 5, 2),
    ("slot_environment", 5, 2),
    # user
    ("user_address_desensitized", 12000, 13),
    ("user_timezone", 120, 7),
    ("user_country_id", 120, 7),
    ("user_state_id", 400, 8),
    ("user_dma_code", 500, 9),
    ("user_platform_os_id", 12, 3),
    ("user_device_type", 10, 3),
    # advertisement (42)
    ("candidate__brand_id", 2000, 11),
    ("candidate__network_id", 16, 4),
    ("candidate__ad_id", 250, 8),
    ("candidate__advertiser_id", 1200, 10),
    ("candidate__duration", 60, 6)
]
numerical_paras = [
    # request
    'slot_width_normalized', 
    'slot_height_normalized', 
    'slot_min_duration_normalized', 
    'slot_max_ad_duration_normalized', 
    # context
    'utc_hour_normalized', 
    'local_hour_normalized', 
    'month_normalized', 
    'day_of_month_normalized', 
    'day_of_week_normalized', 
    'day_of_year_normalized', 
    'week_of_year_normalized'
]


def neighborhood_likelihood_loss(y_true, y_pred):
    # y_true = (label, bidding_price, winning_price
    # y_pred = (price_step)
    y_pred = ops.convert_to_tensor(y_pred)        # (None_all, price_step)
    y_true = math_ops.cast(y_true, y_pred.dtype)  # (None_all, 3)
    
    # arg
    price_min = -3.0
    price_max = 4.0
    price_interval = 0.1
    price_step = tf.cast(tf.shape(y_pred)[-1], tf.int32)
    
    # split y_true
    y_true_label_1d = K.flatten(tf.slice(y_true, [0,0], [-1,1]))  # (None_all,)
    # caculate the bidding price bucket index
    y_true_b = tf.slice(y_true, [0,1], [-1,1])  # (None_all, 1)
    y_true_b = tf.clip_by_value(y_true_b, price_min, price_max)
    y_true_b_idx_2d = tf.cast(tf.floor((y_true_b - price_min) / price_interval), 
                              dtype='int32')  # (None_all, 1)
    y_true_b_idx_1d = K.flatten(y_true_b_idx_2d)  # (None_all,)
    # caculate the winning price bucket index
    y_true_z = tf.slice(y_true, [0,2], [-1,1])  # (None_all, 1)
    y_true_z = tf.clip_by_value(y_true_z, price_min, price_max)
    y_true_z_idx_2d = tf.cast(tf.floor((y_true_z - price_min) / price_interval),
                              dtype='int32')  # (None_all, 1)
    y_true_z_idx_1d = K.flatten(y_true_z_idx_2d)  # (None_all,)
    
    # Calculate masks
    ## on All bids
    mask_win = y_true_label_1d  # (None,)
    mask_lose = 1 - mask_win  # (None,)

    mask_z_cdf = tf.sequence_mask(
                    y_true_z_idx_1d + 1, 
                    price_step)  # (None, price_step)
    mask_z_pdf = tf.math.logical_xor(
                    mask_z_cdf, 
                    tf.sequence_mask(
                        y_true_z_idx_1d,
                        price_step))  # (None, price_step)

    mask_b_cdf = tf.sequence_mask(
                    y_true_b_idx_1d + 1, 
                    price_step)  # (None, price_step)
    mask_b_pdf = tf.math.logical_xor(
                    mask_b_cdf, 
                    tf.sequence_mask(
                        y_true_b_idx_1d, 
                        price_step))  # (None, price_step)
    ## on Winning bids
    mask_win_z_cdf = tf.boolean_mask(mask_z_cdf, mask_win)  # (None_win, price_step)
    mask_win_z_pdf = tf.boolean_mask(mask_z_pdf, mask_win)  # (None_win, price_step)
    mask_win_b_cdf = tf.boolean_mask(mask_b_cdf, mask_win)  # (None_win, price_step)
    mask_win_b_pdf = tf.boolean_mask(mask_b_pdf, mask_win)  # (None_win, price_step)
    ## on Losing bids
    mask_lose_b_cdf = tf.boolean_mask(mask_b_cdf, mask_lose)  # (None_lose, price_step)
    mask_lose_b_pdf = tf.boolean_mask(mask_b_pdf, mask_lose)  # (None_lose, price_step)
    
    # Price Distribution
    y_pred_win = tf.boolean_mask(y_pred, mask_win)  # (None_win, price_step)
    y_pred_lose = tf.boolean_mask(y_pred, mask_lose)  # (None_lose, price_step)
    
    # Loss
    zeros = tf.zeros(tf.shape(y_pred), tf.float32)  # (None, price_step)
    zeros_win = tf.zeros(tf.shape(y_pred_win), tf.float32)  # (None_win, price_step)
    zeros_lose = tf.zeros(tf.shape(y_pred_lose), tf.float32)  # (None_lose, price_step)
    ones = tf.ones(tf.shape(y_pred), tf.float32)  # (None, price_step)
    ones_win = tf.ones(tf.shape(y_pred_win), tf.float32)  # (None_win, price_step)
    ones_lose = tf.ones(tf.shape(y_pred_lose), tf.float32)  # (None_lose, price_step)
    
    # loss_1
    loss_1 = - K.sum(
                tf.math.log(tf.clip_by_value(
                    tf.boolean_mask(
                        y_pred_win,
                        mask_win_z_pdf),
                    K.epsilon(),
                    1.)))
    
    # loss_2_win
    left_neighborhood_offset = y_true_b_idx_1d - y_true_z_idx_1d
    left_neighborhood_idx = tf.math.maximum(y_true_z_idx_1d - left_neighborhood_offset, 0)
    mask_z_neighborhood_cdf = tf.math.logical_xor(
                                    mask_b_cdf, 
                                    tf.sequence_mask(
                                        left_neighborhood_idx,
                                        price_step))
    mask_win_z_neighborhood_cdf = tf.boolean_mask(mask_z_neighborhood_cdf, mask_win)
    loss_2_win = - K.sum(
                    tf.math.log(tf.clip_by_value(
                        K.sum(
                            tf.where(
                                mask_win_z_neighborhood_cdf, 
                                y_pred_win, 
                                zeros_win),
                            axis=1),
                        K.epsilon(),
                        1.)))
    
    # loss_2_lose
    right_neighborhood_offset = 40
    right_neighborhood_idx = tf.math.minimum(y_true_b_idx_1d + right_neighborhood_offset, 
                                             price_step - 1)
    mask_b_neighborhood_cdf = tf.math.logical_xor(
                                    tf.math.logical_not(mask_b_cdf), 
                                    tf.math.logical_not(
                                        tf.sequence_mask(right_neighborhood_idx, 
                                                         price_step)))
    mask_lose_b_neighborhood_cdf = tf.boolean_mask(mask_b_neighborhood_cdf, mask_lose)
    loss_2_lose = - K.sum(
                    tf.math.log(tf.clip_by_value(
                        K.sum(
                            tf.where(
                                mask_lose_b_neighborhood_cdf, 
                                y_pred_lose, 
                                zeros_lose),
                            axis=1),
                        K.epsilon(),
                        1.)))
    
    # loss_2
    beta = 0.8
    loss_2 = beta * loss_2_win + (1 - beta) * loss_2_lose
    
    # total loss
    alpha = 0.2
    return alpha * loss_1 + (1 - alpha) * loss_2


def get_adm_model(learning_rate, l2_reg, price_min, price_max, price_interval, embedd_size):
    price_step = int(math.floor((price_max - price_min + K.epsilon()) / price_interval))
    
    total_len = 0
    input_tensors = []
    
    # Input Embedding Layers
    embedding_tensors = []
    # composing embedding layers
    for column_name, count_value, dimension_length in embedding_paras:
        total_len += embedd_size  # or dimension_length
        input_tensor = Input(name='{}_index'.format(column_name), 
                             shape=(1,), 
                             dtype='int64')
        embedding_tensor = Embedding(
            count_value,
            embedd_size,  # or dimension_length
            input_length=1,
            embeddings_initializer='glorot_normal',
            name='{}_embedding'.format(column_name)
        )(input_tensor)
        embedding_tensor = Flatten()(embedding_tensor)
        embedding_tensors.append(embedding_tensor)
        input_tensors.append(input_tensor)
    
    # Input Numerical Layers
    numerical_tensors = []
    numerical_embedd_tensors = []
    for column_name in numerical_paras:
        total_len += 1
        input_tensor = Input(name='{}'.format(column_name), 
                             shape=(1,), 
                             dtype='float32')
        numerical_tensors.append(input_tensor)
        input_tensors.append(input_tensor)
        input_embedd_tensor = Lambda(lambda x: tf.tile(x, [1, embedd_size]))(input_tensor)
        numerical_embedd_tensors.append(input_embedd_tensor)
    
    feature_num = len(input_tensors)  # 44
    
    # features (embedding + numeric)
    ## 1-order
    x_o1_tensor = Concatenate()(embedding_tensors + numerical_tensors)
    ## 2-order
    x_o2_tensor = Lambda(lambda x: tf.stack(x, axis=1))(embedding_tensors + numerical_embedd_tensors)  # (None, feature_num, dim)
    x_o2_tensor = InteractingLayer(att_embedding_size=8, head_num=2)(x_o2_tensor)
    x_o2_tensor = Flatten()(x_o2_tensor)
    ## high-order
    x_oh_tensor = Dense(total_len/2, 
                        activation='relu', 
                        kernel_regularizer=regularizers.l2(l2_reg))(x_o1_tensor)
    x_oh_tensor = Dense(total_len/4, 
                        activation='relu', 
                        kernel_regularizer=regularizers.l2(l2_reg))(x_oh_tensor)
    
    # output layer
    output_tensor = Concatenate(axis=1)([x_o1_tensor, x_o2_tensor, x_oh_tensor])
    output_tensor = Dense(price_step, 
                          kernel_regularizer=regularizers.l2(l2_reg))(output_tensor)
    output_tensor = Softmax(name='mixture_price_3')(output_tensor)
    
    model = Model(inputs=input_tensors, 
                  outputs=[output_tensor])
    adam = optimizers.Adam(lr=learning_rate)
    optimizer = hvd.DistributedOptimizer(adam)
    model.compile(
        loss=neighborhood_likelihood_loss,
        optimizer=optimizer,
        experimental_run_tf_function=False
    )
    return model