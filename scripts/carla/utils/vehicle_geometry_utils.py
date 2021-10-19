def vehicle_name_to_lf_lr(vehicle_name):
	if vehicle_name   == "vehicle.audi.tt":
		l_f = 1.25 # guesstimated for now.
		l_r = 1.25
	elif vehicle_name == "vehicle.mercedes-benz.coupe":
		l_f = 1.4  # guesstimated for now.
		l_r = 1.4
	else:
		raise NotImplementedError

	return l_f, l_r
