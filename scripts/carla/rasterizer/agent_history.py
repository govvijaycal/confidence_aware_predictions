import carla
import numpy as np
from collections import deque

class ActorInfo:
	""" Class to maintain pose history and bounding box information for a single Carla actor. """
	def __init__(self, actor, history_max_length):		
		self.id = actor.id
		self.type_id = actor.type_id
		self.extent = np.array([actor.bounding_box.extent.x, actor.bounding_box.extent.y])
		
		self.time_history = deque(maxlen=history_max_length)
		self.pose_history = deque(maxlen=history_max_length)

	def get_snapshot(self, index):
		snapshot_dict = {}		
		snapshot_dict['time']     = self.time_history[index]
		snapshot_dict['centroid'] = self.pose_history[index][:2]
		snapshot_dict['yaw']      = self.pose_history[index][2]
		snapshot_dict['extent']   = self.extent
		return snapshot_dict

	def update(self, time, transform):
		""" Add the current time and pose to the history.  
		    We save this in a RHS system vs. Carla's default LHS system.
		"""		
		x_rhs   = transform.location.x
		y_rhs   = -transform.location.y 
		yaw_rhs = -np.radians(transform.rotation.yaw)

		self.time_history.append(time)
		self.pose_history.append([x_rhs, y_rhs, yaw_rhs])

	def prune_outdated_history(self, duration_to_keep=1):
		""" This can be used to manually flush out old data from history, 
			i.e. that are more than duration_to_keep seconds old.
		    Unused as of now due to history_max_length size constraint of deque.
		"""

		earliest_time_to_keep = self.pose_history[-1][0] - duration_to_keep # timestamp (inclusive) after which we will keep history

		num_elements_to_drop  = np.argwhere( np.array(self.time_history) >= earliest_time_to_keep )[0].item()

		for _ in num_elements_to_drop:
			self.time_history.popleft()
			self.pose_history.popleft()

class AgentHistory:
	""" Class to manage pose history for the vehicle/pedestrian agents.
	    Required since snapshots only contain the current agent state and not the past.
	"""
	def __init__(self, actor_list, history_max_length=20):	
		""" Initialize the ActorInfo history for each agent.  
		    NOTE: we assume no agents will be created/destroyed in the episode of interest.
		"""	
		vehicle_actors = actor_list.filter('vehicle*')
		walker_actors  = actor_list.filter('walker*')

		# Identify the ego vehicle.
		ego_vehicles    = [actor for actor in vehicle_actors if actor.attributes['role_name'] == 'hero']
		if len(ego_vehicles) == 0:
			raise RuntimeError("No ego vehicle detected!  Specify a vehicle with role_name hero.")
		elif len(ego_vehicles) > 1:
			raise RuntimeError("Multiple ego vehicles detected! Specify just one vehicle with role_name hero.")
		else:
			pass
		self.ego_vehicle = ActorInfo(ego_vehicles[0], history_max_length)

		# Identify other vehicles in the scene.
		self.npc_vehicles = [ActorInfo(actor, history_max_length) for actor in vehicle_actors if actor.attributes['role_name'] != 'hero']

		# Identify pedestrians in the scene.
		self.pedestrians    = [ActorInfo(actor, history_max_length) for actor in actor_list.filter('*walker*')]

	def update(self, world_snapshot):
		""" Update the state for each relevant actor we are monitoring, given a Carla world snapshot. """
		time = world_snapshot.timestamp.elapsed_seconds

		self.ego_vehicle.update(time, world_snapshot.find(self.ego_vehicle.id).get_transform() )

		for veh in self.npc_vehicles:
			veh.update(time, world_snapshot.find(veh.id).get_transform())

		for ped in self.pedestrians:
			ped.update(time, world_snapshot.find(ped.id).get_transform())

	def query(self, 
		      history_secs = [0.0],  # how many seconds back should we query for each snapshot
		      closeness_eps = 0.1    # how close in seconds is acceptable to produce a valid snapshot
		):
		""" Given a set of relative timestamps looking back, returns a dictionary of "snapshots" encoding the
		    scene over these relative timestamps.  Note history_secs is meant to be nonnegative.
		"""

		snapshots = {}
		tms = np.array(self.ego_vehicle.time_history)
		current_tm = tms[-1]

		for hsec in history_secs:
			tm_query = current_tm - hsec
			ind_closest = np.argmin( np.abs( tms - tm_query ) )
			scene_dict = {}

			if np.abs(tms[ind_closest] - tm_query) <= closeness_eps:				
				scene_dict['ego_vehicle']  = [self.ego_vehicle.get_snapshot(ind_closest)]
				scene_dict['npc_vehicles'] = [veh.get_snapshot(ind_closest) for veh in self.npc_vehicles]
				scene_dict['pedestrians']  = [ped.get_snapshot(ind_closest) for ped in self.pedestrians]
			
			snapshots[np.round(hsec, 2)] = scene_dict

		return snapshots
