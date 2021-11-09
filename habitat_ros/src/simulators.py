# ROS IMPORTS
import rospy
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry

from sensor_callbacks import getPoseMessage
import numpy as np
import habitat
import habitat_sim
from habitat_sim.utils import common as utils
import threading
import tf
import geometry_msgs.msg
import tf2_ros


from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3

class RosEnv(habitat.Env):
  def __init__(self, *args, **kwargs):
    super(RosEnv, self).__init__(*args, **kwargs)


class ContinuousEnvironmentSimulator(threading.Thread):
    """
      Class that emulates a continuous environment.
      Agent can be moved using velocity control and the command self.move().
      Observations have to be extracted in the main thread as otherwise context errors occur.
    """

    def __init__(self, configPath = "/home/rene/catkin_ws/src/habitat-lab/configs/tasks/pointnav_gibson.yaml"):
        super().__init__()
        
        self.lock = threading.Lock() # Lock to either gather observation or move agent
        self.env = habitat.Env(config=habitat.get_config(configPath))
        self.sim = self.env.sim

        self.sim.initialize_agent(agent_id=0)
        self.agent = self.sim.agents[0]

        self.sensor_callbacks = {} # Stores callback for sensor data

        self._agent_gt_pose_pub = rospy.Publisher(
            "~agent_gt_pose", PoseStamped, queue_size=1
        )

        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = False
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True


        self.stop = False
        _x_axis = 0
        _y_axis = 1
        _z_axis = 2
        _dt = 0.00478
        controlRate = 40 # Hz
        self.frame_skip = 2
        self._frame_skip_r = rospy.Rate(controlRate * self.frame_skip)
        self.time_step = 1.0 / (self.frame_skip * controlRate)
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()

        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=50)
        self.world_odom_pub =  rospy.Publisher("odom", Odometry, queue_size=50)
        self.control =  {
                    "forward_velocity": 0,  # [0,2)
                    "rotation_velocity": 0,  # [-1,1)
                    "x" : 0,
                    "y" : 0,
                    "z": 0
                }
        agent_state = self.agent.state
        agent_state.position[0] = 0
        agent_state.position[2] = 0
        self.agent.set_state(agent_state)

    def publishObservations(self):
        """
            Gathers information using sim.get_sensor_observations() and publishes them if a ros publisher exists
            MUST BE CALLED FROM MAIN THREAD, otherwise WEBGL fails due to context errors.
        """
        self.lock.acquire()
        observations = self.sim.get_sensor_observations()
        for sensor in self.sensor_callbacks.keys():

            if "cam" in sensor:
              self.sensor_callbacks[sensor](None)
            else:
              if sensor not in observations.keys():
                  rospy.logwarn_throttle(10, "Did not find observation for registered sensor: %s", sensor)
                  self.sensor_callbacks[sensor]((0*observations["depth"]).astype(np.uint8))
              else:
                  self.sensor_callbacks[sensor](observations[sensor])



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
      # vel = np.array([0, 0, self.control["forward_velocity"]])
      vel = np.array([self.control["x"], self.control["y"], self.control["z"]])

      self.vel_control.linear_velocity.x = vel[1]
      self.vel_control.linear_velocity.y = 0
      self.vel_control.linear_velocity.z = -vel[2]

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

        self.agent_state = agent_state



    def odom_timer(self, evnet):
      from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector
      from habitat_sim import geo

      rotation_mp3d_habitat = quat_from_two_vectors(geo.GRAVITY, np.array([0, 0, -1]))
      p = TransformStamped()
      p.header.stamp = rospy.Time.now()
      p.header.frame_id = "world"
      p.child_frame_id = "map"

      p.transform.translation.x = 0
      p.transform.translation.y = 0
      p.transform.translation.z = 0

      # Make sure the quaternion is valid and normalized
      p.transform.rotation.x = rotation_mp3d_habitat.x
      p.transform.rotation.y = rotation_mp3d_habitat.y
      p.transform.rotation.z = rotation_mp3d_habitat.z
      p.transform.rotation.w = rotation_mp3d_habitat.w

      self.broadcaster.sendTransform(p)

      def getPoseMessageT(position, orientation, frame="map"):
        p = TransformStamped()
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = frame
        p.child_frame_id = "cam"

        # # Make sure the quaternion is valid and normalized
        # from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector
        # from habitat_sim import geo
        #
        # rotation_mp3d_habitat = quat_from_two_vectors(geo.GRAVITY, np.array([0, 0, -1]))
        #
        # # Make sure the quaternion is valid and normalized
        # p.transform.rotation.x = rotation_mp3d_habitat.x
        # p.transform.rotation.y = rotation_mp3d_habitat.y
        # p.transform.rotation.z = rotation_mp3d_habitat.z
        # p.transform.rotation.w = rotation_mp3d_habitat.w
        #
        # position = quat_rotate_vector(rotation_mp3d_habitat, position)
        p.transform.translation.x = position[0]
        p.transform.translation.y = position[1]
        p.transform.translation.z = position[2]

        p.transform.rotation.x = orientation.x
        p.transform.rotation.y = orientation.y
        p.transform.rotation.z = orientation.z
        p.transform.rotation.w = orientation.w
        return p

      self.broadcaster.sendTransform(getPoseMessageT(self.agent_state.position, self.agent_state.rotation))

      p = TransformStamped()
      p.header.stamp = rospy.Time.now()
      p.header.frame_id = "cam"
      p.child_frame_id = "agent_cam"

      p.transform.translation.x = 0
      p.transform.translation.y = 0
      p.transform.translation.z = 0

      # Make sure the quaternion is valid and normalized
      p.transform.rotation.x = 1
      p.transform.rotation.y = 0
      p.transform.rotation.z = 0
      p.transform.rotation.w = 0

      self.broadcaster.sendTransform(p)

      p = TransformStamped()
      p.header.stamp = rospy.Time.now()
      p.header.frame_id = "world"
      p.child_frame_id = "odom"
      p.transform.translation.x = 0
      p.transform.translation.y = 0
      p.transform.translation.z = 0
      # Make sure the quaternion is valid and normalized
      p.transform.rotation.x = 0
      p.transform.rotation.y = 0
      p.transform.rotation.z = 0
      p.transform.rotation.w = 1

      self.broadcaster.sendTransform(p)

      # next, we'll publish the odometry message over ROS
      odom = Odometry()
      odom.header.stamp = rospy.Time.now()
      odom.header.frame_id = "odom"


      # print((self.agent_state.rotation))
      # print((rotation_mp3d_habitat))
      # print(type(self.agent_state.rotation))
      # print(type(rotation_mp3d_habitat))
      import quaternion
      self.agent_state.rotation = quaternion.from_euler_angles(0,0,0.5*3.1415) * rotation_mp3d_habitat * self.agent_state.rotation
      # set the position
      odom.pose.pose = Pose(Point(self.agent_state.position.x,-self.agent_state.position.z, self.agent_state.position.y), Quaternion(
        *[self.agent_state.rotation.x, self.agent_state.rotation.y, self.agent_state.rotation.z, self.agent_state.rotation.w]))




      # set the velocity
      odom.child_frame_id = "base_link"
      odom.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))

      # publish the message
      self.odom_pub.publish(odom)

    def run(self):

        self.timer = rospy.Timer(rospy.Duration(0.1), self.odom_timer)
        """ This ROS thread deals with movement of the robot. """
        while not rospy.is_shutdown():
          self.move()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self ,type, value, traceback):
        self.stop = True
        self.env.__exit__(type, value, traceback)

