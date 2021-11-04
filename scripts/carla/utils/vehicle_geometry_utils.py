def vehicle_name_to_dimensions(vehicle_name):
	if vehicle_name   == "vehicle.audi.tt":
		l_f    = 1.25 # meters, guesstimated for now.
		l_r    = 1.25
		length = 4.2
		width  = 1.8
	elif vehicle_name == "vehicle.mercedes-benz.coupe":
		l_f    = 1.4  # meters, guesstimated for now.
		l_r    = 1.4
		length = 4.7
		width  = 1.8
	else:
		raise NotImplementedError

	return {"lf"     : l_f, \
	        "lr"     : l_r, \
	        "length" : length, \
	        "width"  : width}
