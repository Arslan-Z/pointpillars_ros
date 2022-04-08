#!/usr/bin/env python3
import rospy

from sensor_msgs.point_cloud2 import PointCloud2, PointField
import numpy as np
import os

POINTCLOUD_TOPIC = '/pointcloud' # publishing topic name
SLEEP = 5 # sleep time in s
POINTCLOUDS_PATH = '/home/kitti_original/training/velodyne'
# Set the fixed frame in Rviz to that name to visualize the pointcloud
FRAME_ID_NAME = 'map'

'''
    Reads a file contain pointclouds.bin files and publish them as PointCloud2 msg
'''
class DatasetPlayback:
    def __init__(self):
        # init node
        self.node = rospy.init_node("dataset_node", anonymous=True)
        # publisher
        self.publisher = rospy.Publisher(POINTCLOUD_TOPIC, PointCloud2)
        # pointclouds paths
        paths = sorted(os.listdir(POINTCLOUDS_PATH))
        self.paths = [os.path.join(POINTCLOUDS_PATH, p) for p in paths]

        # publish dataset pointclouds
        self.playback()

    def playback(self):
        '''
            Reads pointclouds from dataset with shape (N, 4) and publishe it
        '''


        for path in self.paths:
            if rospy.is_shutdown():
                return

            pointcloud = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

            pointcloud_msg = self.convert_numpy_to_ros_pointcloud(pointcloud)
            # pointcloud_msg.header.frame_id = 'id_' + str(i)

            rospy.loginfo("playback publisher with points " + str(pointcloud.shape))
            self.publisher.publish(pointcloud_msg)

            rospy.sleep(SLEEP)

    @staticmethod
    def convert_numpy_to_ros_pointcloud(points):
        '''
            Convert numpy pointcloud of shape (N, 4) to pointcloud msg to be published
        '''
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header.frame_id = FRAME_ID_NAME
        pointcloud_msg.header.stamp = rospy.Time.now()

        pointcloud_msg.height = 1
        pointcloud_msg.width = len(points)

        # XYZI fields
        pointcloud_msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)]

        pointcloud_msg.is_bigendian = False
        # its 4 bytes * 4 elements[XYZI]
        pointcloud_msg.point_step = 16
        pointcloud_msg.row_step = pointcloud_msg.point_step * points.shape[0]
        pointcloud_msg.is_dense = False
        pointcloud_msg.data = np.asarray(points, np.float32).tostring()
        return pointcloud_msg

if __name__ == "__main__":
    try:
        DatasetPlayback()
    except rospy.ROSInterruptException:
        exit()

