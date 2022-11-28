"""Write ETH data into TFRecord files


 Args:
        1, 2, 3: year, month, day       of first radar file to be read
        4, 5, 6: year, month, day       of last radar file to be read.
        7: directory of radar data
        8: directory of ETH data
        9: directory for TFR files
        example: 2009 1 1 2009 12 31 $HOME/radar_data $HOME/ETH_data $HOME/TFR_data

        Saves
        'image': image array  ,0s if data does not exist
        'image_sum': float sum of image
        'exists': boolean int if data exists
        'date': time stamp of image
         for every eth data file of a single day in a TFRecord file
        """
#### feature include negative does not mean image should not be safed. It means that if this is first frame,
# the window will not be included. However frame can still be part of different windows

import pathlib
import pandas as pd
import h5py
import numpy as np
import tensorflow as tf
from pandas.tseries.offsets import DateOffset
import sys
import math
rng = np.random.default_rng(17)

start = pd.Timestamp(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
end = pd.Timestamp(int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
data_dir_radar = pathlib.Path(sys.argv[7])
data_dir_eth = pathlib.Path(sys.argv[8])
out_dir = pathlib.Path(sys.argv[9])
ISS_value = 200
# The lower the Importance Sampling Scheme value the more frames will be included


print(start)
print(end)
print(data_dir_radar)
print(data_dir_eth)
print(out_dir)
print("ISS value", ISS_value)

def check_eth_h5_file(file):
    file_okay = 0
    try:
        f = h5py.File(file, 'r')
        image = f['image1']['image_data']
        image = np.asarray(image)
        x = image != 255
        if np.any(x):
            file_okay = 1
        f.close()
    except:
        pass

    return file_okay

def check_radar_h5_file(file):
    file_okay = 0
    try:
        f = h5py.File(file, 'r')
        image = f['image1']['image_data']
        image = np.asarray(image)
        x = image != 65535
        if np.any(x):
            file_okay = 1
        f.close()
    except:
        pass

    return file_okay


def check_eth_h5_file(file):
  file_okay = 0
  try:
    f = h5py.File(file, 'r')
    image = f['image1']['image_data']
    image = np.asarray(image)
    x = image != 255
    if np.any(x):
      file_okay = 1
    f.close()
  except:
    pass

  return file_okay

def prepare_eth_h5_file(file, upper_row=300, lowest_row=556, left_column=241, right_column=497 ):
    f = h5py.File(file, 'r')
    image = f['image1']['image_data']
    image = np.asarray(image, dtype=np.float32)
    f.close()
    image = image[upper_row:lowest_row, left_column:right_column]
    # missing value == 0
    image = np.where(image == 0, -1.0, image)
     # out of image == 255
     # changed from NaN to 0
    image = np.where(image == 255, 0.0, image * 0.062992)

    if (np.isnan(image).any()):
        print(file, "contains nan values in ETH")

    return image


def prepare_radar_h5_file(file, upper_row=300, lowest_row=556, left_column=241, right_column=497 ):
    f = h5py.File(file, 'r')
    image = f['image1']['image_data']
    image = np.asarray(image, dtype=np.float32)
    f.close()
    image = image[upper_row:lowest_row, left_column:right_column]
    image = np.where( image == 65535, np.NaN, ((image / 100.0) * 12) )
    image = np.where(image > 1024, 1024, image)

    global Max_val_r
    if np.nanmax(image) > Max_val_r:
      Max_val_r = np.nanmax(image)

    image_sum = np.sum(image)

    if (np.isnan(image).any()):
        print(file, "contains nan values in radar")

    return image, image_sum

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _array_feature(array):
  serialized_array= tf.io.serialize_tensor(array)
  return _bytes_feature(serialized_array)

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def image_example(image_radar, image_eth, image_sum_radar, status_radar, status_both, include, event_sum, date):
  feature = {
      'image_radar': _array_feature(image_radar),
      'image_eth': _array_feature(image_eth),
      'image_sum_radar': _float_feature(image_sum_radar),
      'exists_radar': _int64_feature(status_radar),
      'exists_both': _int64_feature(status_both),
      'include_if_first_frame' : _int64_feature(include),
      'event_sum' : _float_feature(event_sum),
      'date':_bytes_feature(date)
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))

Max_val_r = 0
def write_day_TRfile(df, file_path):
  if df.shape[0] != 12 * 24:
    raise AssertionError("Dataframe contains {r} instead of 12*24 rows".format(r=df.shape[0]))
  count_not_included = 0
  tf_record_options = tf.io.TFRecordOptions(compression_type="GZIP")
  with tf.io.TFRecordWriter(file_path, options=tf_record_options) as writer:
    frames = []
    # create value dictionary for every frame
    for index in range(12 * 24):
      row = df.iloc[index, :]
      status_radar = int(row['file_radar_okay'])
      status_both = int(row['both_files_okay'])
      date = row['date']

      # add image according to status
      if bool(status_both):
        image_eth = prepare_eth_h5_file(row['file_eth_observed'])
        image_radar, image_sum_radar = prepare_radar_h5_file(row['file_radar_observed'])
      elif bool(status_radar):
        print(row['date'], "has invalid ETH data")
        image_radar, image_sum_radar = prepare_radar_h5_file(row['file_radar_observed'])
        image_eth = np.full([1, 1], 0, dtype=np.float32)
      else:
        print(row['date'], "has invalid radar (and possibly ETH) data")
        image_eth = np.full([1, 1], 0, dtype=np.float32)
        image_radar = np.full([1, 1], 0, dtype=np.float32)
        image_sum_radar = 0.0

      example = {'image_radar': image_radar, "image_sum_radar": image_sum_radar, "image_eth": image_eth,
                 'status_radar': status_radar, 'status_both': status_both, 'date': date}
      frames.append(example)

    # 0 dont include
    # 1 include
    # 2 include but ETH not available
    # go through all frames to calculate rolling average of following 21 frames
    for index in range(12 * 24):
      example = frames[index]
      event_sum = 0
      # if both ETH and radar or only radar data exist:
      if bool(example['status_radar']) :
        include = 1
        # if not 22 frames can be gathered
        if index > 266:
          include = 0
          count_not_included += 1

        # if index small enough
        else:
        # add sum of following 21 frames
          event_sum = 0
          for i in range(22):
            event_sum += frames[index+i]['image_sum_radar']
            # if future radar is invalid
            if bool(frames[index+i]['status_radar']) == False:
              include = 0
              break
            # if radar data okay but ETH not
            if bool(example['status_both']) == False:
                include = 2
          prob = 1 - math.exp(-(event_sum / (ISS_value* 12)))
          prob = min(1.0, prob + 0.3)
          if prob < rng.random():
            include = 0
            count_not_included += 1
      # if no radar image
      else:
        include = 0

      example_message = image_example(image_radar=example['image_radar'], image_eth=example['image_eth'],
                                        image_sum_radar=example['image_sum_radar'],
                                        status_radar=example['status_radar'],
                                        status_both=example['status_both'],
                                        include= include,
                                        event_sum = event_sum,
                                        date=example['date'].encode())

      writer.write(example_message.SerializeToString())

  return count_not_included

def create_tfrecord_files(start,end):

    current_date = start
    total_not_included = 0
    day_count = 0
    while current_date <= end:
        day_count += 1
        print(current_date)
        following_date = current_date + DateOffset(1)
        year, month, day = current_date.strftime('%Y'), current_date.strftime('%m'), current_date.strftime('%d')
        file_names_radar = list(data_dir_radar.glob('{y}/{m}/*{y}{m}{d}*.h5'.format(y=year,m=month, d=day)))
        file_names_eth = list(data_dir_eth.glob('{y}/{m}/*{y}{m}{d}*.h5'.format(y=year,m=month, d=day)))

        files_radar_df = pd.DataFrame({'file_radar': file_names_radar})
        files_radar_df['name_radar'] = files_radar_df['file_radar'].apply(lambda path: path.name)
        files_eth_df = pd.DataFrame({'file_eth': file_names_eth})
        files_eth_df['name_eth'] = files_eth_df['file_eth'].apply(lambda path: path.name)

        expected_dates = [dt.strftime('%Y%m%d%H%M') for dt in pd.date_range(start=current_date, end=following_date, freq="300s")][:-1]
        expected_dates_df = pd.DataFrame({'date': expected_dates})
        expected_dates_df['file_eth'] = expected_dates_df['date'].apply(
            lambda date: data_dir_eth / date[4:6] / ('RAD_NL25_ETH_NA_' + date + '.h5'))
        expected_dates_df['file_radar'] = expected_dates_df['date'].apply(
            lambda date: data_dir_radar / date[4:6] / ('RAD_NL25_RAC_5min_' + date + '_cor.h5'))

        expected_dates_df['name_radar'] = expected_dates_df['file_radar'].apply(lambda path: path.name)
        expected_dates_df['name_eth'] = expected_dates_df['file_eth'].apply(lambda path: path.name)


        merged_df = pd.merge(expected_dates_df, files_radar_df, how='left', left_on='name_radar', right_on='name_radar',
                             suffixes=('_expected', '_observed'))
        merged_df = pd.merge(merged_df, files_eth_df, how='left', left_on='name_eth', right_on='name_eth',
                             suffixes=('_expected', '_observed'))

        print(pd.isna(merged_df['file_radar_observed']).sum(), "radar files not found")
        print(pd.isna(merged_df['file_eth_observed']).sum(), "eth files not found")
        merged_df['file_radar_okay'] = merged_df['file_radar_observed'].apply(check_radar_h5_file)
        merged_df['file_eth_okay'] = merged_df['file_eth_observed'].apply(check_eth_h5_file)
        print((merged_df['file_radar_okay'] == 0).sum(), "radar files not valid")
        print((merged_df['file_eth_okay'] == 0).sum(), "eth files not valid")
        merged_df['both_files_okay'] =  merged_df['file_radar_okay'] * merged_df['file_eth_okay']
        TFfile_path = out_dir / "{y}/{m}/".format(y=year,m=month)
        TFfile_path.mkdir(parents=True, exist_ok=True)
        shard_name = TFfile_path / "Joined_RAD_NL25_ETH_NA_{y}{m}{d}.tfrecords".format(y=year,m=month,d=day)
        not_included = write_day_TRfile(merged_df, str(shard_name))
        print(not_included, "frames have negative ISS value")
        total_not_included += not_included
        current_date = following_date
        print(Max_val_r)
    p = int((total_not_included / (day_count * 288)) * 100)
    print("In total {} of {} frames have been assigned negative ISS ({} percent)".format(total_not_included,
                                                                                       day_count*288,p ))
    end_of_day_frames = 21 * day_count
    percentage = (end_of_day_frames  / total_not_included) * 100
    print("{} or {} perecent of these frames are negative due to too short windows".format(end_of_day_frames, int(percentage)))
    print("The highest rainfall rate was {} mm/h". format(Max_val_r))



create_tfrecord_files(start,end)

# TODO also calculate average and sd or max for normalizing
