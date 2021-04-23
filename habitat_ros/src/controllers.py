# ROS IMPORTS
from habitat_ros_utils.srv import PublishTransform
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_callbacks import getPoseMessage
from sensor_msgs.msg import CameraInfo
from nav_msgs.msg import Odometry
import math
import numpy as np
import habitat
import habitat_sim
from habitat_sim.utils import common as utils
import threading
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
import quaternion

# import sys
# sys.path = [
#     b for b in sys.path if "2.7" not in b
# ]  # remove path's related to ROS from environment or else certain packages like cv2 can't be imported
#
# import tf

class RosEnv(habitat.Env):
  def __init__(self, *args, **kwargs):
    super(RosEnv, self).__init__(*args, **kwargs)


class ContinuousEnvironmentSimulator(threading.Thread):
    """
      Class that emulates a continuous environment.
      Agent can be moved using velocity control and the command self.move().
      Observations have to be extracted in the main thread as otherwise context errors occur.
    """

    def __init__(self, configPath = "/home/rene/catkin_ws/src/active_learning_for_segmentation/habitat_ros/src/configs/tasks/imagenav_gibson.yaml"):
        super().__init__()
        
        self.lock = threading.Lock() # Lock to either gather observation or move agent
        # self.env = habitat.Env(config=habitat.get_config(configPath))

        config = habitat.get_config(config_paths='/home/rene/catkin_ws/src/active_learning_for_segmentation/habitat_ros/src/configs/datasets/pointnav/gibson.yaml')
        config.defrost()
        config.DATASET.DATA_PATH = '/home/rene/catkin_ws/src/active_learning_for_segmentation/habitat_ros/src/data/datasets/gibson/v1/val/val.json.gz'
        config.DATASET.SCENES_DIR = '/home/rene/catkin_ws/src/active_learning_for_segmentation/habitat_ros/src/data/scene_datasets/'
        config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR']
        config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 256
        config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 256
        config.SIMULATOR.TURN_ANGLE = 30
        config.freeze()


        self.env = habitat.Env(config=config)

        self.sim = self.env.sim

        self.sim.initialize_agent(agent_id=0)
        self.agent = self.sim.agents[0]

        self.sensor_callbacks = {} # Stores callback for sensor data

        self._agent_gt_pose_pub = rospy.Publisher(
            "~agent_gt_pose", PoseStamped, queue_size=1
        )
        self._agent_odom_pub = rospy.Publisher(
            "~agent_gt_odom", Odometry, queue_size=10
        )

        self._tf_service = rospy.ServiceProxy("/habitat_ros/publishTransform", PublishTransform)

        self._camera_infos = []

        for name, entry in self.sim.sensor_suite.sensors.items():
          print(name, entry)
          print(entry.sensor_type)
          if entry.sensor_type == habitat.SensorTypes.COLOR or \
            entry.sensor_type == habitat.SensorTypes.DEPTH or \
            entry.sensor_type == habitat.SensorTypes.SEMANTIC:

            camera_info = CameraInfo()
            camera_info.header.frame_id = name
            camera_info.width = entry.config.WIDTH
            camera_info.height = entry.config.HEIGHT
            cx = camera_info.width / 2.0
            cy = camera_info.height / 2.0
            fx = camera_info.width / (
                2.0 * math.tan(entry.config.HFOV * math.pi / 360.0))
            fy = fx
            camera_info.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            camera_info.D = [0, 0, 0, 0, 0]
            camera_info.R = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
            camera_info.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1.0, 0]

            self._camera_infos.append(
              { "publisher":  rospy.Publisher("~agent/cam/" + name + "/cameraInfo", CameraInfo, queue_size=10),
              "cameraInfo": camera_info
              })

        # self.odom_broadcaster = tf.TransformBroadcaster()

        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True


        self.stop = False
        _x_axis = 0
        _y_axis = 1
        _z_axis = 2
        _dt = 0.00478
        controlRate = 100 # Hz
        self.frame_skip = 2
        self._frame_skip_r = rospy.Rate(controlRate * self.frame_skip)
        self.time_step = 1.0 / (self.frame_skip * controlRate)

        self.control =  {
                    "forward_velocity": 0,  # [0,2)
                    "rotation_velocity": 0,  # [-1,1)
                }

    def publishObservations(self):
        """
            Gathers information using sim.get_sensor_observations() and publishes them if a ros publisher exists
            MUST BE CALLED FROM MAIN THREAD, otherwise WEBGL fails due to context errors.
        """
        self.lock.acquire()
        observations = self.sim.get_sensor_observations()
        for sensor in self.sensor_callbacks.keys():
            if sensor not in observations.keys():
                rospy.logwarn("Did not find observation for registered sensor: %s", sensor)
            else:
              ts = rospy.Time.now()
              self.sensor_callbacks[sensor](observations[sensor], ts)
              if(sensor == "rgb"):
                camera_pub = self._camera_infos[0]
                info = camera_pub['cameraInfo']
                info.header = CameraInfo().header
                info.header.frame_id = "rgb"
                info.header.stamp = ts
                camera_pub['publisher'].publish(info)
                print("published info ")

        # print("cam,")
        # for camera_pub in self._camera_infos:
        #   info = camera_pub['cameraInfo']
        #   info.header = CameraInfo().header
        #   camera_pub['publisher'].publish(info)
        #   print("published info ")

        self.lock.release()

    def addSensorCallback(self, sensorId, callbackFn):
      """
      Adds a callback function for a given sensor
      Args:
        sensor_id: The sensor id, which is defined in the habitat env. config
        callback_fn: The callback function which will be called with the result of the sensor observation

      Returns: Void

      """
      self.sensor_callbacks[sensorId] = callbackFn

    def move(self):
      """ Moves based on velocity commands specified in self.control['forward_velocity'] and self.control["rotation_velocity"]
      Code from here https://colab.research.google.com/github/facebookresearch/habitat-sim/blob/master/examples/tutorials/colabs/ECCV_2020_Navigation.ipynb#scrollTo=TrdI1S_vshsM
      """

      # local forward is -z
      vel = np.array([0, 0, -self.control["forward_velocity"]])
      self.vel_control.linear_velocity.x = vel[0]
      self.vel_control.linear_velocity.y = vel[1]
      self.vel_control.linear_velocity.z = vel[2]

      # local up is y
      rot = np.array([0, self.control["rotation_velocity"], 0])
      self.vel_control.angular_velocity.x = rot[0]
      self.vel_control.angular_velocity.y = rot[1]
      self.vel_control.angular_velocity.z = rot[2]

      for _frame in range(self.frame_skip):
        self._frame_skip_r.sleep()
        self.lock.acquire()
        # Integrate the velocity and apply the transform.
        # Note: this can be done at a higher frequency for more accuracy
        agent_state = self.agent.state
        previous_rigid_state = habitat_sim.RigidState(
          utils.quat_to_magnum(agent_state.rotation), agent_state.position
        )

        # manually integrate the rigid state
        target_rigid_state = self.vel_control.integrate_transform(
          self.time_step, previous_rigid_state
        )

        # snap rigid state to navmesh and set state to object/agent
        # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
        end_pos = self.sim.step_filter(
          previous_rigid_state.translation, target_rigid_state.translation
        )

        # set the computed state
        agent_state.position = end_pos
        agent_state.rotation = utils.quat_from_magnum(
          target_rigid_state.rotation
        )
        self.agent.set_state(agent_state)

        # Check if a collision occured
        dist_moved_before_filter = (
            target_rigid_state.translation - previous_rigid_state.translation
        ).dot()
        dist_moved_after_filter = (
            end_pos - previous_rigid_state.translation
        ).dot()

        # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
        # collision _didn't_ happen. One such case is going up stairs.  Instead,
        # we check to see if the the amount moved after the application of the filter
        # is _less_ than the amount moved before the application of the filter
        EPS = 1e-5
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
        # do something with it if wanted

        # run any dynamics simulation
        self.sim.step_physics(self.time_step)
        self.lock.release()

        odomMsg = Odometry()
        odomMsg.header.frame_id = "odom"
        odomMsg.pose.pose = Pose(Point(agent_state.position.x,agent_state.position.y,agent_state.position.z),
                                 Quaternion(agent_state.rotation.x,agent_state.rotation.y,agent_state.rotation.z,agent_state.rotation.w))

        odomMsg.twist.twist.linear.x = self.vel_control.linear_velocity.x
        odomMsg.twist.twist.linear.y = self.vel_control.linear_velocity.y
        odomMsg.twist.twist.linear.z = self.vel_control.linear_velocity.z

        odomMsg.twist.twist.angular.x = self.vel_control.angular_velocity.x
        odomMsg.twist.twist.angular.y = self.vel_control.angular_velocity.y
        odomMsg.twist.twist.angular.z = self.vel_control.angular_velocity.z

        self._agent_gt_pose_pub.publish(getPoseMessage(agent_state.position, agent_state.rotation))
        self._agent_odom_pub.publish(odomMsg)

        transform = PublishTransform()
        transform.tx = agent_state.position.x
        transform.ty = agent_state.position.y
        transform.tz = agent_state.position.z

        transform.rx = agent_state.rotation.x
        transform.ry = agent_state.rotation.y
        transform.rz = agent_state.rotation.z
        transform.rw = agent_state.rotation.w

        transform.child_id = "odom"
        transform.parent_id = "world_agent"

        agent_rotation_world =   agent_state.rotation * quaternion.quaternion(-0.7071,-0.7071,0,0)
        self._tf_service.call(0,0,0, \
            agent_rotation_world.x, agent_rotation_world.y, agent_rotation_world.z, agent_rotation_world.w, "agent", "odom")


        self._tf_service.call(0,0,0, 0,1,0,0, "camera", "agent")

        self._tf_service.call(0,0,0, 0,-0.707,0,0.707, "rgb", "camera")



        self._tf_service.call(-agent_state.position.z, -agent_state.position.x,-agent_state.position.y, \
            0,0,0,1, "odom", "world")


        # self._tf_service.call(0,0,0, \
        #     -0.7071,-0.7071,0,0, "world_agent", "world")

    # agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z, agent_state.rotation.w
    def run(self):
        """ This ROS thread deals with movement of the robot. """
        while not rospy.is_shutdown():
          self.move()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self ,type, value, traceback):
        self.stop = True
        self.env.__exit__(type, value, traceback)

