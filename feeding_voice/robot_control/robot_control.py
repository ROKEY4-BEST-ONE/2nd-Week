import os
import time
import sys
from scipy.spatial.transform import Rotation
import numpy as np
import rclpy
from rclpy.node import Node
import DR_init
from od_msg.srv import SrvDepthPosition
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
from robot_control.onrobot import RG
package_path = get_package_share_directory("pick_and_place_voice")
# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"
DEPTH_OFFSET = -5.0
MIN_DEPTH = 2.0
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
rclpy.init()
dsr_node = rclpy.create_node("robot_control_node", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node
try:
    from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, trans
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()
########### Gripper Setup. Do not modify this area ############
gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
########### Robot Controller ############
class RobotController(Node):
    def __init__(self):
        super().__init__("feeding_voice")
        # :둥근_압핀: 이 작업에서는 미리 정의된 좌표 데이터베이스가 필요 없습니다.
        # self.pose_database = { ... }
        self.init_robot()
        # Service Clients
        self.get_position_client = self.create_client(SrvDepthPosition, "/get_3d_position")
        self.get_keyword_client = self.create_client(Trigger, "/get_keyword")
        # Wait for services
        if not self.get_position_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("get_3d_position 서비스를 찾을 수 없습니다.")
            sys.exit()
        if not self.get_keyword_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("get_keyword 서비스를 찾을 수 없습니다.")
            sys.exit()
        self.get_logger().info("모든 서비스가 연결되었습니다.")
        self.get_position_request = SrvDepthPosition.Request()
        self.get_keyword_request = Trigger.Request()
    def robot_control(self):
        self.get_logger().info("="*50)
        self.get_logger().info("음성 명령을 기다립니다. 'Hello Rokey'라고 말하고 명령하세요.")
        get_keyword_future = self.get_keyword_client.call_async(self.get_keyword_request)
        rclpy.spin_until_future_complete(self, get_keyword_future)
        if get_keyword_future.result() and get_keyword_future.result().success:
            response_message = get_keyword_future.result().message
            self.get_logger().info(f"음성 인식 키워드: '{response_message}'")
            try:
                tools_str, destinations_str = response_message.split('/')
                object_to_pick = tools_str.strip().split()[0]
                place_to_go_object = destinations_str.strip().split()[0]
            except (ValueError, IndexError):
                self.get_logger().error(f"잘못된 형식의 응답입니다: '{response_message}'. '도구 / 목적지' 형식이 필요합니다.")
                return
            # :둥근_압핀: 1. Pick 위치 (카메라로 '사과' 찾기)
            self.get_logger().info(f"--- Pick 대상 '{object_to_pick}' 위치 찾는 중 ---")
            pick_pos = self.get_object_position_from_camera(object_to_pick)
            if pick_pos is None:
                self.get_logger().error(f"'{object_to_pick}'을(를) 찾지 못해 작업을 중단합니다.")
                return
            # :둥근_압핀: 2. Place 위치 (카메라로 '입술' 찾기)
            self.get_logger().info(f"--- Place 대상 '{place_to_go_object}' 위치 찾는 중 ---")
            place_pos = self.get_object_position_from_camera(place_to_go_object)
            if place_pos is None:
                self.get_logger().error(f"'{place_to_go_object}'을(를) 찾지 못해 작업을 중단합니다.")
                return
            # 3. Pick and Place 실행
            self.get_logger().info(f"'{object_to_pick}'을(를) 집어 '{place_to_go_object}'로 옮깁니다.")
            self.pick_and_place_target(pick_pos, place_pos)
            self.init_robot()
        else:
            self.get_logger().warn("키워드 인식에 실패했거나 서비스 호출에 실패했습니다.")
    def get_object_position_from_camera(self, target_name):
        self.get_position_request.target = target_name
        self.get_logger().info(f"객체 인식 노드에 '{target_name}'의 3D 위치를 요청합니다.")
        get_position_future = self.get_position_client.call_async(self.get_position_request)
        rclpy.spin_until_future_complete(self, get_position_future)
        if get_position_future.result():
            result = get_position_future.result().depth_position.tolist()
            if sum(result) == 0:
                self.get_logger().warn(f"카메라가 '{target_name}'을(를) 찾지 못했습니다.")
                return None
            self.get_logger().info(f"카메라 좌표 수신: {result}")
            gripper2cam_path = os.path.join(package_path, "resource", "T_gripper2camera.npy")
            robot_posx = get_current_posx()[0]
            td_coord = self.transform_to_base(result, gripper2cam_path, robot_posx)
            if td_coord[2] and sum(td_coord) != 0:
                td_coord[2] += DEPTH_OFFSET
                td_coord[2] = max(td_coord[2], MIN_DEPTH)
            target_pos = list(td_coord[:3]) + robot_posx[3:]
            return target_pos
        return None
    def init_robot(self):
        JReady = [0, 0, 90, 0, 90, 0]
        self.get_logger().info("준비 자세로 이동합니다.")
        movej(JReady, vel=VELOCITY, acc=ACC)
        gripper.open_gripper()
        mwait()
    def pick_and_place_target(self, pick_pos, place_pos):
        self.get_logger().info(f"Pick 위치로 이동: {pick_pos[:3]}")
        movel(pick_pos, vel=VELOCITY, acc=ACC)
        mwait()
        gripper.close_gripper()
        mwait()
        self.get_logger().info("Pick 완료.")
        self.get_logger().info(f"Place 위치로 이동: {place_pos[:3]}")
        # :둥근_압핀: 안전을 위해 입술 근처 Z축에 오프셋을 둘 수 있습니다.
        # place_pos_safe = place_pos[:]
        # place_pos_safe[2] += 50.0 # Z축으로 5cm 위까지만 접근
        # movel(place_pos_safe, vel=VELOCITY, acc=ACC)
        movel(place_pos, vel=VELOCITY, acc=ACC)
        mwait()
        self.get_logger().info("Place 완료. 그리퍼를 엽니다.")
        gripper.open_gripper()
        mwait()
    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T
    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1)
        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)
        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)
        return td_coord[:3]
def main(args=None):
    node = RobotController()
    try:
        while rclpy.ok():
            node.robot_control()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == "__main__":
    main()
