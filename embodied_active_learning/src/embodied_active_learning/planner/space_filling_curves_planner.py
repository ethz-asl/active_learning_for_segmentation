#!/usr/bin/env python3
"""
"""
# ros
import tf
import rospy
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
import random
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from geometry_msgs.msg import Transform, Quaternion, Point, Twist
import math
import pickle
import numpy as np


class SpaceFillingCurvesPlanner:
    """
    """

    def __init__(self):
        '''  Initialize ros node and read params '''
        with open(r"/home/rene/catkin_ws/src/active_learning_for_segmentation/embodied_active_learning/scripts/trajectory_evaluation/trajectory.pickle", "rb") as output_file:
            self.trajectory = pickle.load( output_file)

        self.trajectoryIdx = 0
        self.lastYaw = 0
        minimapPath = rospy.get_param("/minimap/path", None)
        if minimapPath is None:
            rospy.logerr("[MinimapManager] Minimap path was not specified!")
            return

        try:
            map_struct = pickle.load(open(minimapPath, "rb"))
        except Exception as e:
            rospy.logerr("[MinimapManager] Could not open minimap {}".format(
                str(e)))

        self.x = 0
        self.running = False
        self.forceReplan = True
        self.y = 0

        self.map_rgb = map_struct['map'].astype(np.uint8)

        if len(self.map_rgb.shape) == 2 or self.map_rgb.shape[-1] == 1:
            rospy.loginfo("Provided map was only one channel")
            self.map_rgb = (map_struct['map'] > 0).astype(np.uint8)
            # Convert to 3 channel image
            self.map_rgb = np.stack(
                [(map_struct['map'] == 0) * 255 for _ in range(3)],
                axis=-1).astype(np.uint8)

        self.lengthPerPixel = map_struct['dimensions'][
            'lengthPerPixel']  # Conversion from pixel to meters in unreal
        self.top_start, self.left_start = map_struct['start'][
                                              'top'], map_struct['start']['left']  # Start position of the drone
        self.top_start = self.top_start - 30
        self.left_start = self.left_start + 20
        self.top_lim, self.left_lim = self.map_rgb.shape[0:2]

        self._start_service = rospy.Service("planner_node/toggle_running", SetBool, self.toggleRunning)
        self._odom_cb = rospy.Subscriber("odometry", Odometry, self.callback)
        self._collision_sub = rospy.Subscriber("/airsim/drone_1/collision", Bool, self.collisionCallback)

        self._trajectory_pub = rospy.Publisher("command/trajectory", MultiDOFJointTrajectory)
        rospy.loginfo("BumpAndRotatePlanner running")

    def toggleRunning(self, req):
        self.running = req.data
        print("toggled running", req.data)
        return True, 'running'

    def publishGoal(self):
        msg = MultiDOFJointTrajectory()
        msg.header.frame_id = "world"
        q = tf.transformations.quaternion_from_euler(0, 0, self.lastYaw)
        self.lastYaw += 3.1415 / 10
        if self.lastYaw >= 2 * math.pi:
            self.lastYaw = -math.pi

        nextPoint = self.trajectory[self.trajectoryIdx]
        self.trajectoryIdx += 1
        print("x = (", nextPoint[1] , "-", self.left_start , ")*", self.lengthPerPixel)
        print("y ? (", nextPoint[0] , "-", self.top_start , ")*", self.lengthPerPixel)
        self.x = ( nextPoint[1]- self.left_start) * self.lengthPerPixel
        self.y = ( nextPoint[0]- self.top_start )  * self.lengthPerPixel
        z = 0

        print("NEXT GOAL: {},{} (top,left) {},{}".format(self.x, self.y, nextPoint[0], nextPoint[1]))
        transforms = Transform(translation=Point(self.x,self.y,z),
                               rotation=Quaternion(q[0], q[1], q[2], q[3]))
        msg.points.append(MultiDOFJointTrajectoryPoint([transforms], [Twist()], [Twist()], rospy.Time(0)))
        self._trajectory_pub.publish(msg)

    def collisionCallback(self, collisionMessage, step=0.3):
        if not self.running:
            return
        print("[COLLIDED].")
        # self.publishGoal()

    def callback(self, odomMsg):
        if not self.running:
            return
        self.currentPose = odomMsg.pose.pose
        x = self.currentPose.position.x
        y = self.currentPose.position.y
        q = self.currentPose.orientation

        if self.forceReplan or abs(y - self.y) + abs(x - self.x) < 0.1:
            self.publishGoal()
            print("[MOVING] Going to publish next goal {:.4f},{:.4f},{:.4f}".format(self.x,
                                                                                    self.y,
                                                                                    self.lastYaw))
            self.forceReplan = False

if __name__ == '__main__':
    rospy.init_node('space_filling_curves_planner', anonymous=True)
    planner = SpaceFillingCurvesPlanner()
    rospy.spin()