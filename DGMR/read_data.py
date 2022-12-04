import pathlib
import tensorflow as tf
import functools
import numpy as np

seed = 18
tf.random.set_seed(seed)

feature_description = {
    'image_radar': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image_eth': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image_sum_radar': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'exists_radar': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'exists_both': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'include_if_first_frame': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'event_sum':tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'date': tf.io.FixedLenFeature([], tf.string, default_value=''),
}


def _parse_function_radar(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    example['image_radar'] = tf.io.parse_tensor(example['image_radar'], tf.float32, name=None)
    #example['image_radar'] = min_max_normalize(example['image_radar'], 0, 1024)
    example.pop('image_eth')
    example.pop('exists_both')
    return example

def _parse_function_ETH(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    example['image_radar'] = tf.io.parse_tensor(example['image_radar'], tf.float32, name=None)
    #example['image_radar'] = min_max_normalize(example['image_radar'], 0, 1024)
    example['image_eth'] = tf.io.parse_tensor(example['image_eth'], tf.float32, name=None)
    #example['image_eth'] = min_max_normalize(example['image_eth'], 0.2, 16)
    return example

def min_max_normalize(image, min_val, max_val):
    image = tf.clip_by_value(image, -1, max_val)
    image_norm = (image - min_val) / (max_val - min_val)
    return image_norm

def only_image_radar(image_dataset):
    return image_dataset['image_radar']

def only_image_ETH(image_dataset):
    return zip(image_dataset['image_radar'], image_dataset['image_eth'])

def check_dates(image_dataset):
    return zip(image_dataset['date'], image_dataset['date'])


def filter_windows_old(d, expected_len, ISS=True, ISS_value=200):
    window_ok = True
    if len(d['date']) != expected_len:
        window_ok = False
    for example in d['exists_radar']:
        # the second column is whether the image is ok
        if example == 0:
            window_ok = False
            break
    # Importance Sampling Scheme
    # stochastically filter out sequences that contain little rainfall
    if ISS:
        rain_sum = 0.0
        for element in d['image_sum_radar']:
            rain_sum += element
        prob = 1 - tf.math.exp(-(rain_sum / ((ISS_value* 12)/1000)))
        print(prob)
        prob = tf.math.minimum(1.0, prob + 0.1)
        if prob < tf.random.uniform(shape=[]):
            window_ok = False
    return window_ok


def filter_windows(d, expected_len = 22, ETH = False, ISS=True, ISS_value = 200, base_prob = 0.3):
    window_ok = True
    if len(d['date']) != expected_len:
        window_ok = False
    # If ISS has standard values just read out 'include_if_first_frame'
    if ISS and ISS_value == 200 and base_prob== 0.3:
        for x in d['include_if_first_frame']:
            if x == 0:
                window_ok = False
            # if ETH data relevant
            if x ==2 and ETH:
                window_ok = False
            break
    else:
        if len(d['date']) != expected_len:
            window_ok = False
        if ETH:
            for example in d['exists_both']:
                if example == 0:
                    window_ok = False
        else:
            for example in d['exists_radar']:
                if example == 0:
                    window_ok = False
        # If ISS has not standard values calculate ISS probability
        if ISS:
            event_sum = 0.0
            for x in d['event_sum']:
                event_sum = x
                break
            prob = 1 - tf.math.exp(-(event_sum /(ISS_value * 12)))
            prob = tf.math.minimum(1.0, prob + base_prob)
            if prob < tf.random.uniform(shape=[]):
                window_ok = False
    return window_ok

def read_TFR(path, ETH= False, batch_size=12, window_shift=1, ISS = True, ISS_value=200, check=False):
    window_size = 22
    if ISS_value != 200:
        tf.print("ISS value different from 200 means windows will be manually filtered")
    tfr_dir = pathlib.Path(path)
    pattern = str(tfr_dir / '*/*/*.tfrecords')
    dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(pattern, seed=seed), compression_type='GZIP')
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    dataset = dataset.with_options(options)
    ## dataset = shards.interleave(tf.data.TFRecordDataset)
    if ETH:
        dataset = dataset.map(_parse_function_ETH, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(_parse_function_radar, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.window(size=window_size, shift=window_shift)
    filter_function = functools.partial(filter_windows, expected_len=window_size, ETH=ETH, ISS=ISS, ISS_value=ISS_value)
    dataset = dataset.filter(filter_function)
    if check:
        dataset = dataset.map(check_dates, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        if ETH:
            dataset = dataset.map(only_image_ETH, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(only_image_radar, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


#------------------------------------------------------------------------------------#

def only_image_radar_test(image_dataset):
    return zip(image_dataset['image_radar'], image_dataset['date'])

def only_image_ETH_test(image_dataset):
    return zip(image_dataset['image_radar'], image_dataset['image_eth'],image_dataset['date'])

def test_TFR(path, ETH= False, batch_size=12, window_shift=1, ISS = True, ISS_value=200, check=False):
    window_size = 22
    if ISS_value != 200:
        tf.print("ISS value different from 200 means windows will be manually filtered")
    tfr_dir = pathlib.Path(path)
    pattern = str(tfr_dir / '*/*/*.tfrecords')
    dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(pattern,shuffle=False, seed=seed), compression_type='GZIP',num_parallel_reads = 1 )
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    dataset = dataset.with_options(options)
    ## dataset = shards.interleave(tf.data.TFRecordDataset)
    if ETH:
        dataset = dataset.map(_parse_function_ETH, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(_parse_function_radar, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.window(size=window_size, shift=window_shift)
    filter_function = functools.partial(filter_windows, expected_len=window_size, ETH=ETH, ISS=ISS, ISS_value=ISS_value)
    dataset = dataset.filter(filter_function)
    if check:
        dataset = dataset.map(check_dates, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        if ETH:
            dataset = dataset.map(only_image_ETH_test, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(only_image_radar_test, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    #dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset



def read_TFR_noshuffel(path, ETH= False, batch_size=12, window_shift=1, ISS = True, ISS_value=200, check=False):
    window_size = 22
    if ISS_value != 200:
        tf.print("ISS value different from 200 means windows will be manually filtered")
    tfr_dir = pathlib.Path(path)
    pattern = str(tfr_dir / '*/*/*.tfrecords')
    dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(pattern, seed=seed), compression_type='GZIP')
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    dataset = dataset.with_options(options)
    ## dataset = shards.interleave(tf.data.TFRecordDataset)
    if ETH:
        dataset = dataset.map(_parse_function_ETH, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(_parse_function_radar, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.window(size=window_size, shift=window_shift)
    filter_function = functools.partial(filter_windows, expected_len=window_size, ETH=ETH, ISS=ISS, ISS_value=ISS_value)
    dataset = dataset.filter(filter_function)
    if check:
        dataset = dataset.map(check_dates, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        if ETH:
            dataset = dataset.map(only_image_ETH, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(only_image_radar, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset