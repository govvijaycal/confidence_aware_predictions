import tensorflow as tf
import numpy as np
from multiprocessing import Process

##############################
# Standard TFRecord feature generation from https://www.tensorflow.org/tutorials/load_data/tfrecord.
def _bytes_feature_list(value):
  if isinstance(value,type(tf.constant(0))):
    value = value.numpy()
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

##############################
# Write trajectory dataset to tfrecord format.
def write_tfrecord(file_prefix,           # where to save the tfrecord without the extension
	               dataset,               # list of dataset entries to write
	               data_dict_function,    # function to convert entry to a dictionary
	               shuffle = False,       # whether to shuffle the data_list before writing
	               shuffle_seed = 0,      # set a random seed for deterministic shuffling
	               max_per_record = None, # whether to impose a max constraint on items per record	               
	               ):
	
	# Remove the suffix if given in the file_prefix string.
	if '.record' in file_prefix:
		file_prefix = file_prefix.split('.record')[0]
	if '.tfrecord' in file_prefix:
		file_prefix = file_prefix.split('.tfrecord')[0]		

	# Shuffle the dataset if required with a specified seed.
	if shuffle:
		np.random.seed(shuffle_seed)
		np.random.shuffle(dataset)
	num_elements = len(dataset)

	# Handle case where there are a max number of items per record.
	# Else just write all items to one record.
	if max_per_record:
		assert type(max_per_record) == int
	else:
		max_per_record = num_elements

	current_split = 0
	current_dataset_ind = 0
	total_entries_written = 0

	while current_dataset_ind < num_elements:
		record_file = '{}_{}.record'.format(file_prefix, current_split)
		writer = tf.io.TFRecordWriter(record_file)
		print('***Started writing to {} at dataset index {} of {}'.format(
			   record_file, current_dataset_ind, num_elements))

		for ind in range(max_per_record):			
			data_dict = data_dict_function(dataset[current_dataset_ind])

			if data_dict is not None and len(data_dict.keys()) > 0:
				# We have a valid entry, let's write to the record.
				ftr = {}
				for key in ['instance', 'sample', 'type']:
					ftr[key] = _bytes_feature_list(tf.compat.as_bytes(data_dict[key]))

				for key in ['velocity', 'acceleration', 'yaw_rate']:
					ftr[key] = _bytes_feature_list( tf.io.serialize_tensor(np.float64(data_dict[key])) )

				for key in ['pose', 'past_poses_local', 'future_poses_local']:
					ftr[key] = _bytes_feature_list( tf.io.serialize_tensor(data_dict[key].astype(np.float64)) )
					ftr[key + '_shape'] = \
					    _bytes_feature_list( np.array(data_dict[key].shape, np.int32).tobytes() ) 

				ftr['image']       = \
				    _bytes_feature_list( data_dict['image'].tobytes() ) 
				ftr['image_shape'] = \
				    _bytes_feature_list( np.array(data_dict['image'].shape, np.int32).tobytes() ) 

				example = tf.train.Example(features = tf.train.Features(feature=ftr))
				writer.write(example.SerializeToString())
				total_entries_written += 1
			else:
				# We did not write a valid entry so let's not increment ind (number written).
				# We still increment current_dataset_ind below to move on to the next dataset entry.
				ind -= 1

			current_dataset_ind += 1
			if current_dataset_ind == num_elements:
				break

		writer.close()
		current_split += 1		

	print('Finished writing: {} splits, {} entries written out of {}'.format(
		   current_split, total_entries_written, num_elements))

def _parse_function(proto):
	ftr = {'instance'                 : tf.io.FixedLenFeature([], tf.string),
	       'sample'                   : tf.io.FixedLenFeature([], tf.string),
	       'type'                     : tf.io.FixedLenFeature([], tf.string),
	       'velocity'                 : tf.io.FixedLenFeature([], tf.string),
	       'acceleration'             : tf.io.FixedLenFeature([], tf.string),
	       'yaw_rate'                 : tf.io.FixedLenFeature([], tf.string),
	       'pose'                     : tf.io.FixedLenFeature([], tf.string),
	       'pose_shape'               : tf.io.FixedLenFeature([], tf.string),
	       'past_poses_local'         : tf.io.FixedLenFeature([], tf.string),
	       'past_poses_local_shape'   : tf.io.FixedLenFeature([], tf.string),
	       'future_poses_local'       : tf.io.FixedLenFeature([], tf.string),
	       'future_poses_local_shape' : tf.io.FixedLenFeature([], tf.string),
	       'image'                    : tf.io.FixedLenFeature([], tf.string),
	       'image_shape'              : tf.io.FixedLenFeature([], tf.string)
	      }

	data_dict = {}
	
	parsed_features = tf.io.parse_single_example(proto, ftr)

	for key in ['instance', 'sample', 'type']:
		data_dict[key] = parsed_features[key]

	for key in ['velocity', 'acceleration', 'yaw_rate']:
		data_dict[key] = tf.reshape(tf.io.parse_tensor(parsed_features[key], out_type=tf.float64), (1,))


	for key in ['pose', 'past_poses_local', 'future_poses_local']:
		value   = tf.io.parse_tensor(parsed_features[key], out_type=tf.float64)
		shape = tf.io.decode_raw(parsed_features[key + '_shape'], tf.int32) 
		data_dict[key] = tf.reshape(value, shape)

	image              = tf.io.decode_raw(parsed_features['image'], tf.uint8)
	image_shape        = tf.io.decode_raw(parsed_features['image_shape'], tf.int32)
	data_dict['image'] = tf.reshape(image, image_shape)

	return data_dict








	
