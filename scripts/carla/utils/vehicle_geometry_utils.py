def vehicle_name_to_dimensions(vehicle_name):
	if vehicle_name   == "vehicle.audi.tt":
		l_f    = 1.25  # meters, guesstimated for now.
		l_r    = 1.25  # meters, guesstimated for now.
		length = 4.181 # meters (Carla BBox)
		width  = 1.994 # meters (Carla BBox)
	elif vehicle_name == "vehicle.mercedes-benz.coupe":
		l_f    = 1.4   # meters, guesstimated for now.
		l_r    = 1.4   # meters, guesstimated for now.
		length = 5.027   # meters (Carla BBox)
		width  = 2.152   # meters (Carla BBox)
	else:
		raise NotImplementedError

	return {"lf"     : l_f, \
	        "lr"     : l_r, \
	        "length" : length, \
	        "width"  : width}
