import os
import glob

datadir = os.path.abspath(__file__).split('scripts')[0] + 'data'

L5KIT_TRAIN = [ f"{datadir}/l5kit_train_{x}.record" for x in range(36) ]
L5KIT_VAL   = [ f"{datadir}/l5kit_train_{x}.record" for x in range(36, 45) ]
L5KIT_TEST  = [ f"{datadir}/l5kit_val_{x}.record" for x in range(9) ]

NUSCENES_TRAIN = [ f"{datadir}/nuscenes_train_{x}.record" for x in range(33) ]
NUSCENES_VAL   = [ f"{datadir}/nuscenes_train_val_{x}.record" for x in range(9) ]
NUSCENES_TEST  = [ f"{datadir}/nuscenes_val_{x}.record" for x in range(10) ]

if __name__ == "__main__":
	l5kit_records    = glob.glob( f"{datadir}/l5kit*.record" )
	nuscenes_records = glob.glob( f"{datadir}/nuscenes*.record")

	# Check there exists data in the specified location.
	assert len(l5kit_records) > 0
	assert len(nuscenes_records) > 0

	# Check that splits comprise all identified tfrecords.	
	assert len(l5kit_records) == (len(L5KIT_TRAIN) + len(L5KIT_VAL) + len(L5KIT_TEST))
	assert len(nuscenes_records) == (len(NUSCENES_TRAIN) + len(NUSCENES_VAL) + len(NUSCENES_TEST))

	# Check that splits are disjoint.
	assert set(L5KIT_TRAIN).intersection( set(L5KIT_VAL) )  == set()
	assert set(L5KIT_TRAIN).intersection( set(L5KIT_TEST) ) == set()
	assert set(L5KIT_TEST).intersection( set(L5KIT_VAL) )   == set()
	
	assert set(NUSCENES_TRAIN).intersection( set(NUSCENES_VAL) )  == set()
	assert set(NUSCENES_TRAIN).intersection( set(NUSCENES_TEST) ) == set()
	assert set(NUSCENES_TEST).intersection( set(NUSCENES_VAL) )   == set()

	# Print out number of records in each dataset/split.
	print("L5KIT:")
	for split, split_str in zip([L5KIT_TRAIN, L5KIT_VAL, L5KIT_TEST], \
	                            ["train", "val", "test"]):
		print(f"\t{split_str}: {len(split)} records")

	print("NUSCENES:")
	for split, split_str in zip([NUSCENES_TRAIN, NUSCENES_VAL, NUSCENES_TEST], \
	                            ["train", "val", "test"]):
		print(f"\t{split_str}: {len(split)} records")
