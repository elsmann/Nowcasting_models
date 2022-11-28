import pathlib
import tensorflow as tf
import functools

seed = 17
tf.random.set_seed(seed)

feature_description = {
    'image_radar': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image_eth': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image_sum_radar': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'exists_radar': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'exists_both': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'date': tf.io.FixedLenFeature([], tf.string, default_value=''),
}


def _parse_function_ETH(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example

## add false if time later then 21 frames before midnight
# rolling sum lower then iss
# radar or both don't exsitits of one in the frame


# save for entries for whole day in array
# look at following 21 indexes for filtering


# TODO add if date == first of month, if difference between dates bigger than 5 min?
def filter_windows(d, expected_len, ISS=True, ISS_value=200):
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
        prob = 1 - tf.math.exp(-(rain_sum / ISS_value))
        prob = tf.math.minimum(1.0, prob + 0.1)
        if prob < tf.random.uniform(shape=[]):
            window_ok = False
    return window_ok



def read_TFR(path, ETH= False, batch_size=32, window_shift=1, ISS=200):
    tfr_dir = pathlib.Path(path)
    pattern = str(tfr_dir / '*/*/*.tfrecords')
    dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(pattern, seed=seed))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    dataset = dataset.with_options(options)
    dataset = dataset.map(_parse_function_ETH, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.window(size=22, shift=window_shift)
    filter_function = functools.partial(filter_windows, expected_len=22, ISS = ISS)
    dataset = dataset.filter(filter_function)