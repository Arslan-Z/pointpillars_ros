#!/usr/bin/env python3
import rospy
from sensor_msgs.point_cloud2 import PointCloud2, read_points
from geometry_msgs.msg import Vector3, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

import sys,math, time
sys.path.append('../')
from pathlib import Path
import numpy as np
import torch
from google.protobuf import text_format
import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, second_builder)
from second.pytorch.train import example_convert_to_torch, predict_kitti_to_anno
from second.core import box_np_ops

# ================================================
class BBox3D:
    def __init__(self, x, y, z, h, w, l, rotation):
        """
            3D BBox in LIDAR Coordiantes
        """
        self.pos = (x,y,z)
        self.dims = (h,w,l)
        self.x = x
        self.y = y
        self.z = z
        self.height = h # z length 20
        self.width  = w # y length 10
        self.length = l # x length 50
        self.rotation = rotation
class LabelObject:
    def __init__(self, bbox_3d, label, score=1):
        self.bbox_3d = bbox_3d
        self.label = label
        self.score = score
# ================================================

POINTCLOUD_TOPIC = '/pointcloud'

class Pointpillars:
    def __init__(self,
        config_path='/home/nutonomy_pointpillars/second/configs/pointpillars/car/xyres_16.proto',
            model_dir='/path/to/model_dir',
            checkpoint='/home/kitti_original/voxelnet-117309.tckpt'
        ):
        self.score_threshold = 0.3
        # Configs
        model_dir = str(Path(model_dir).resolve())
        if isinstance(config_path, str):
            config = pipeline_pb2.TrainEvalPipelineConfig()
            with open(config_path, "r") as f:
                proto_str = f.read()
                text_format.Merge(proto_str, config)
        else:
            config = config_path

        input_cfg = config.eval_input_reader
        self.model_cfg = config.model.second
        self.class_names = list(input_cfg.class_names)
        self.center_limit_range = self.model_cfg.post_center_limit_range
        #########################
        # Build Voxel Generator & Target Assigner
        self.voxel_generator = voxel_builder.build(self.model_cfg.voxel_generator)
        bv_range = self.voxel_generator.point_cloud_range[[0, 1, 3, 4]]
        box_coder = box_coder_builder.build(self.model_cfg.box_coder)
        target_assigner_cfg = self.model_cfg.target_assigner
        self.target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                        bv_range, box_coder)
        #########################
        # Build Voxelnet Pointpillars
        self.net = second_builder.build(self.model_cfg, self.voxel_generator, self.target_assigner, 1)
        self.net.cuda()
        #########################
        # Load model weights
        if checkpoint is None:
            torchplus.train.try_restore_latest_checkpoints(model_dir, [self.net])
        else:
            torchplus.train.restore(checkpoint, self.net)
        self.net.eval()
        #########################
        # ROS Subsctibers & Pubishers
        self.init_ros()
    
    def init_ros(self):
        # init node
        self.node = rospy.init_node("pointpillars_node", anonymous=True)
        # publisher & subscriber formats
        self.pointcloud_subscriber = rospy.Subscriber(POINTCLOUD_TOPIC, PointCloud2, self.callback)
        self.rviz_boxes_publisher = rospy.Publisher('/visualization_marker_array', MarkerArray)
        self.prev_predictions_rviz = 0
        self.rviz_pointcloud_publisher = rospy.Publisher('/rviz_pointcloud', PointCloud2)
        # keep the python script from terminating to receive published pointclouds
        rospy.spin()

    @staticmethod
    def convert_pointcloud2_to_numpy(pointcloud_msg:PointCloud2):
        points = read_points(pointcloud_msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
        pointcloud = np.zeros((pointcloud_msg.width, 4), np.float32)
        for i, point in enumerate(points):
            pointcloud[i] = np.array([point[0],point[1],point[2], point[3]])
        return pointcloud

    # callback for pointcloud subscriber
    def callback(self, pointcloud_msg:PointCloud2):
        rospy.loginfo("Reveived pointcloud with points " + str(pointcloud_msg.width * pointcloud_msg.height))

        t1 = time.time()
        pointcloud = self.convert_pointcloud2_to_numpy(pointcloud_msg)
        t2 = time.time()
        input_example = self.preprocess_pointcloud(pointcloud)
        t3 = time.time()

        with torch.no_grad():
            predictions = predict_kitti_to_anno(
                self.net, input_example, self.class_names, self.center_limit_range,
                self.model_cfg.lidar_input, None)

            t4 = time.time()
            predictions = self.pointpillars_output_to_label_objects(predictions)
            t5 = time.time()

            print('ros to numpy = ', t2-t1)
            print('preprocessing = ', t3-t2)
            print('model inference = ', t4-t3)
            print('total = ', t4-t1, ' ... ', t4-t2)
            print('==================')

            # ========== RViz visualization ==========
            # delete all the markers
            if self.prev_predictions_rviz > 0:
                delete_marker_array = MarkerArray()
                for i in range(self.prev_predictions_rviz):
                    delete_marker = Marker()
                    delete_marker.id = i
                    delete_marker.action = Marker.DELETE
                    delete_marker_array.markers.append(delete_marker)
                self.rviz_boxes_publisher.publish(delete_marker_array)

            # Draw the new markers
            rviz_boxes = MarkerArray()
            for id, pred in enumerate(predictions):
                rviz_box = self.convert_prediciton_to_rviz_box(pred)
                rviz_box.id = id
                rviz_box.ns = 'pointpllars_namespace'

                rviz_boxes.markers.append(rviz_box)

            self.rviz_boxes_publisher.publish(rviz_boxes)
            self.rviz_pointcloud_publisher.publish(pointcloud_msg)
            print("predicted ", len(predictions), ' cars')
            self.prev_predictions_rviz = len(predictions)

    @staticmethod
    def convert_prediciton_to_rviz_box(pred:LabelObject):
        rviz_box = Marker()
        box = pred.bbox_3d
        rviz_box.pose.position.x = box.x
        rviz_box.pose.position.y = box.y
        rviz_box.pose.position.z = box.z                
        rviz_box.scale.x = box.width
        rviz_box.scale.y = box.length
        rviz_box.scale.z = box.height
        # ref https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        rviz_box.pose.orientation.x = 0
        rviz_box.pose.orientation.y = 0
        rviz_box.pose.orientation.z = math.sin(box.rotation/2)
        rviz_box.pose.orientation.w = math.cos(box.rotation/2)

        rviz_box.type = Marker.CUBE
        rviz_box.action = Marker.ADD
        rviz_box.header.stamp = rospy.Time.now()
        rviz_box.header.frame_id = 'map'
        rviz_box.color.r = 1.0
        rviz_box.color.b = 0.0
        rviz_box.color.g = 0.0
        rviz_box.color.a = 0.1
        return rviz_box

    '''
        Converts predictions of the pointpillars to label objects in KittiUtils
        Filters the predictions based on the defined score threshold
    '''
    def pointpillars_output_to_label_objects(self, predictions):
        predictions = predictions[0]
        n = len(predictions['name'])

        kitti_objects = []
        for i in range(n):
            bbox = predictions['bbox'][i]
            dims = predictions['dimensions'][i]
            location = predictions['location'][i]
            rotation = predictions['rotation_y'][i]

            # z coord is center in one coordinate and bottom in the other
            location[2] -= location[2]/2

            score = predictions['score'][i]
            if score < self.score_threshold:
                continue

            box3d = BBox3D(location[0], location[1], location[2], dims[1], dims[2], dims[0], -rotation)
            kitti_object = LabelObject(box3d, 1, score)

            kitti_objects.append(kitti_object)
        return kitti_objects

    def preprocess_pointcloud(self, pointcloud):
            pointcloud = pointcloud.reshape(-1,4)

            # [0, -40, -3, 70.4, 40, 1]
            voxel_size = self.voxel_generator.voxel_size # pillar size
            pc_range = self.voxel_generator.point_cloud_range # clip pointcloud ranges
            grid_size = self.voxel_generator.grid_size # ground size 
            max_voxels = 20000
            anchor_area_threshold = 1 
            out_size_factor =2 # rpn first downsample stride / rpn first upsample stride

            feature_map_size = grid_size[:2] // out_size_factor
            feature_map_size = [*feature_map_size, 1][::-1]
            # [352, 400]

            voxels, coordinates, num_points = self.voxel_generator.generate(
                pointcloud, max_voxels)

            # Anchors from target assigner
            ret = self.target_assigner.generate_anchors(feature_map_size)
            anchors = ret["anchors"]
            anchors = anchors.reshape([-1, 7])
            anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
                anchors[:, [0, 1, 3, 4, 6]])
            coors = coordinates
            dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
                coors, tuple(grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchors_area = box_np_ops.fused_get_anchors_area(
                dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
            anchors_mask = anchors_area > anchor_area_threshold

            example = {
                'voxels': voxels,
                'num_points': num_points,
                'coordinates': coordinates,
                'rect': np.identity(4, dtype=np.float32),
                'Trv2c': np.identity(4, dtype=np.float32),
                'P2': np.identity(4, dtype=np.float32),
                "anchors": anchors,
                'anchors_mask': anchors_mask,
                'image_idx': torch.tensor([0]),
                'image_shape': torch.tensor([1242, 35]).reshape((1,2))
            }

            #  [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect'
            #  4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask'
            #  8: 'image_idx', 9: 'image_shape']

            coordinates = np.pad(
                    coordinates, 
                    ((0, 0), (1, 0)),
                    mode='constant',
                    constant_values=1)
            example['coordinates'] = torch.from_numpy(coordinates)

            example = example_convert_to_torch(example, torch.float32)

            example['voxels'] = torch.squeeze(example['voxels'], 0)
            example['num_points'] = torch.squeeze(example['num_points'], 0)
            example['anchors'] = example['anchors'].unsqueeze(0)
            example['anchors_mask'] = example['anchors_mask'].unsqueeze(0)
            example['Trv2c'] = example['Trv2c'].unsqueeze(0)
            example['rect'] = example['rect'].unsqueeze(0)
            example['P2'] = example['P2'].unsqueeze(0)

            example_tuple = list(example.values())

            return example_tuple

if __name__ == "__main__":
    try:
        Pointpillars()
    except rospy.ROSInterruptException:
        exit()

