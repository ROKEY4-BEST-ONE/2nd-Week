import os
import time
import sys
import math
from scipy.spatial.transform import Rotation
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import DR_init
from hh_od_msg.action import TrackLips
from hh_od_msg.srv import SrvDepthPosition, SrvRiceRichPosition, SrvCheckStop
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
from robot_control.onrobot import RG
package_path = get_package_share_directory("feeding_voice")
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
    from DSR_ROBOT2 import movej, movel, movesx, amovel, get_current_posx, mwait, check_force_condition, set_singularity_handling, \
                        DR_VAR_VEL, DR_BASE, DR_AXIS_X, DR_AXIS_Y, DR_AXIS_Z, DR_MV_MOD_REL, DR_FC_MOD_REL, \
                        task_compliance_ctrl, set_desired_force, release_force, release_compliance_ctrl, posx
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()
########### Gripper Setup. Do not modify this area ############
gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
################# Singularity Handling Setup ##################
set_singularity_handling(DR_VAR_VEL)
########### Robot Controller ############
class RobotController(Node):
    def __init__(self):
        super().__init__("feeding_voice")
        self.init_robot()

        # Service Clients
        self.track_lips_action_client       = ActionClient(self, TrackLips, 'track_lips')
        self.get_rice_rich_position_client  = self.create_client(SrvRiceRichPosition, '/get_rice_rich_position')
        self.get_position_client            = self.create_client(SrvDepthPosition, "/get_3d_position")
        self.get_keyword_client             = self.create_client(Trigger, "/get_keyword")

        # Service Servers
        self.check_stop_service             = self.create_service(SrvCheckStop, "/check_stop", self.check_stop_callback)

        # Wait for services
        if not self.get_position_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("get_3d_position 서비스를 찾을 수 없습니다.")
            sys.exit()
        if not self.get_keyword_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("get_keyword 서비스를 찾을 수 없습니다.")
            sys.exit()
        self.get_logger().info("모든 서비스가 연결되었습니다.")

        # Current Tool
        self.tool = None
        self.tool_original_pose = None

        # Define Requests
        self.get_position_request           = SrvDepthPosition.Request()
        self.get_rice_rich_position_request = SrvRiceRichPosition.Request()
        self.get_keyword_request            = Trigger.Request()

    def robot_control(self):
        self.get_logger().info("="*50)
        self.get_logger().info("음성 명령을 기다립니다. 'Hello Rokey'라고 말하고 명령하세요.")
        get_keyword_future = self.get_keyword_client.call_async(self.get_keyword_request)
        rclpy.spin_until_future_complete(self, get_keyword_future)
        if get_keyword_future.result() and get_keyword_future.result().success:
            response_message = get_keyword_future.result().message
            self.get_logger().info(f"음성 인식 키워드: '{response_message}'")
            try:
                food_to_eat = response_message.strip()
            except (ValueError, IndexError):
                self.get_logger().error(f"잘못된 형식의 응답입니다: '{response_message}'.")
                return
            
            # 1. Pick 위치 (카메라로 숟가락 or 포크 찾기)
            pick_tool_success = self.pick_tool_if_needed(food_to_eat)
            if not pick_tool_success:
                return
            
            # 2. Pick 위치 (카메라로 '사과' 찾기)
            pick_food_success = self.pick_food(food_to_eat)
            if not pick_food_success:
                return
            
            # 3. 먹이기 (카메라로 '입술' 찾기)
            self.feed_food()
            self.check_eating()

            # 4. 포지션 복귀
            self.replace_tool()
        else:
            self.get_logger().warn("키워드 인식에 실패했거나 서비스 호출에 실패했습니다.")
    
    def pick_tool_if_needed(self, food_to_eat):
        if food_to_eat == 'rice' or food_to_eat == 'Croissant':
            self.ready_to_pick_tool()
            tool_name = 'spoon' if food_to_eat == 'rice' else 'fork'
            if food_to_eat == 'rice':
                pick_tool_pos, pick_tool_angle = self.get_midpoint_of_two_objects('pororo', 'loopy')
            else:
                pick_tool_pos, pick_tool_angle = self.get_midpoint_of_two_objects('Bulbasaur', 'pikachu')

            if pick_tool_pos is None:
                self.get_logger().error(f"'{tool_name}'을(를) 찾지 못해 작업을 중단합니다.")
                self.init_robot()
                return False
            
            amovel(pick_tool_pos, vel=VELOCITY, acc=ACC)
            movej([0, 0, 0, 0, 0, pick_tool_angle], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
            mwait()
            movel([0, 0, -70, 0, 0, 0], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
            gripper.close_gripper()
            while gripper.get_status()[0]:
                time.sleep(0.1)

            self.tool = tool_name
            self.tool_original_pose = get_current_posx()[0]

            movel([0, 0, 110, 0, 0, 0], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)

            if food_to_eat == 'rice':   # find rice pose
                movej([10, 0, 90, 0, 90, -170], vel=VELOCITY, acc=ACC)
            else:                       # find Croissant pose
                movej([0, 0, 90, 0, 90, -90], vel=VELOCITY, acc=ACC)

        return True
    def pick_food(self, target):
        # 1. Get position of food
        pick_pos = self.get_object_position_from_camera(target)
        if pick_pos is None:
            ## TODO 숟가락이나 포크를 들고 있는 상태라면 가져다 둬야 함.
            self.get_logger().error(f"'{target}'을(를) 찾지 못해 작업을 중단합니다.")
            self.replace_tool()
            return False
        
        # 2. Pick food
        if target == 'rice':
            movel([0, 0, 0, -90, -60, 90], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
            cur_pos = get_current_posx()[0]
            movel(pick_pos[:3] + cur_pos[3:], vel=VELOCITY, acc=ACC)
            movej([0, 0, 0, 0, 0, -90], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
            time.sleep(1000)
            self.detecting(5)
            # movel([0, 0, 2, 0, 0, 0], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
            # pos1 = posx([0, 160, -50, -90, -50, 90])
            # pos2 = posx([0, 40, -10, -90, -10, 90])
            # movesx([pos1, pos2], vel=VELOCITY//3, acc=ACC//3, mod=DR_MV_MOD_REL)
        elif target == 'Croissant':
            movel(pick_pos, vel=VELOCITY, acc=ACC)
            self.detecting()
            time.sleep(0.3)
            movel([0, 0, 70, 0, 0, 0], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
        else: # apple
            self.get_logger().info(f"Pick 위치로 이동: {pick_pos[:3]}")
            movel(pick_pos, vel=VELOCITY, acc=ACC)
            mwait()
            gripper.close_gripper()
            while gripper.get_status()[0]:
                time.sleep(0.1)
            self.get_logger().info("Pick 완료.")
            JReady = [0, 0, 90, 0, 90, -90]
            movej(JReady, vel=VELOCITY, acc=ACC)

        return True
    def feed_food(self):
        self.ready_to_feed_robot()
        self.get_logger().info("입술 추적 시작...")

        self.track_lips_action_client.wait_for_server()
        goal_msg = TrackLips.Goal()
        send_goal_future = self.track_lips_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.track_lips_feedback_callback
        )
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("TrackLips Goal이 거부되었습니다.")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
    def check_eating(self):
        self.get_logger().info("✅ 입술 근접 완료! check_eating 수행")
        while True:
            fc_cond_x = check_force_condition(DR_AXIS_X, min=20)
            fc_cond_y = check_force_condition(DR_AXIS_Y, min=20)
            fc_cond_z = check_force_condition(DR_AXIS_Z, min=20)
            if max(fc_cond_x, fc_cond_y, fc_cond_z) == 0:
                self.get_logger().info('외력 감지됨.')
                time.sleep(3)
                break
            time.sleep(0.1)

    def replace_tool(self):
        if self.tool and self.tool_original_pose:
            tool_upper_pose = self.tool_original_pose.copy()
            tool_upper_pose[2] += 110
            movel(tool_upper_pose, vel=VELOCITY, acc=ACC)
            movel(self.tool_original_pose, vel=VELOCITY, acc=ACC)
            gripper.open_gripper()
            while gripper.get_status()[0]:
                time.sleep(0.1)
            movel(tool_upper_pose, vel=VELOCITY, acc=ACC)
            self.tool = None
            self.tool_original_pose = None
        self.init_robot()

    def check_stop_callback(self, request, response):
        coords = request.coords
        gripper2cam_path = os.path.join(package_path, "resource", "T_gripper2camera.npy")
        robot_posx = get_current_posx()[0]
        td_coord = self.transform_to_base(coords, gripper2cam_path, robot_posx)
        if td_coord[2] and sum(td_coord) != 0:
            if self.tool:
                td_coord[1] += 180.0
            else:
                td_coord[1] += 50.0
            td_coord[2] -= 100.0
            td_coord[2] = max(td_coord[2], MIN_DEPTH)

        dist = math.dist(td_coord[:3], robot_posx[:3])
        self.get_logger().info(f'현재 거리는 {dist} mm 입니다.')
        if dist < 30.0:  # 30mm 이하 → 추적 종료
            response.stop = True
        else:
            response.stop = False
        return response
    def track_lips_feedback_callback(self, feedback_msg):
        depth_pos = feedback_msg.feedback.depth_position
        if len(depth_pos) < 3:
            return
        gripper2cam_path = os.path.join(package_path, "resource", "T_gripper2camera.npy")
        robot_posx = get_current_posx()[0]
        td_coord = self.transform_to_base(depth_pos, gripper2cam_path, robot_posx)
        if td_coord[2] and sum(td_coord) != 0:
            if self.tool:
                td_coord[1] += 180.0
            else:
                td_coord[1] += 50.0
            td_coord[2] -= 100.0
            td_coord[2] = max(td_coord[2], MIN_DEPTH)
        target_pos = list(td_coord[:3]) + robot_posx[3:]

        dist = math.dist(target_pos[:3], robot_posx[:3])
        if dist < 5.0:  # 5mm 이하 → 이동 생략
            return

        self.get_logger().info(f"[즉시 이동] {target_pos} (거리 {dist:.2f}mm)")

        try:
            amovel(target_pos, vel=VELOCITY, acc=ACC)
        except Exception as e:
            self.get_logger().error(f"이동 중 오류: {e}")

    def get_object_position_from_camera(self, target_name):
        if target_name == 'rice':
            self.get_logger().info(f"객체 인식 노드에 'rice'의 3D 위치를 요청합니다.")
            get_rice_rich_position_future = self.get_rice_rich_position_client.call_async(self.get_rice_rich_position_request)
            rclpy.spin_until_future_complete(self, get_rice_rich_position_future)
            if get_rice_rich_position_future.result():
                result = get_rice_rich_position_future.result().depth_position.tolist()
                if sum(result) == 0:
                    self.get_logger().warn(f"카메라가 'rice'를 찾지 못했습니다.")
                    return None
                self.get_logger().info(f"카메라 좌표 수신: {result}")
                gripper2cam_path = os.path.join(package_path, "resource", "T_gripper2camera.npy")
                robot_posx = get_current_posx()[0]
                td_coord = self.transform_to_base(result, gripper2cam_path, robot_posx)
                if td_coord[2] and sum(td_coord) != 0:
                    td_coord[2] = max(td_coord[2], MIN_DEPTH)
                target_pos = list(td_coord[:3]) + robot_posx[3:]
                target_pos[0] += -90.0
                target_pos[1] += 240.0
                target_pos[2] += 0.0
                return target_pos
        else:
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
                    td_coord[2] = max(td_coord[2], MIN_DEPTH)
                target_pos = list(td_coord[:3]) + robot_posx[3:]
                if target_name == 'Croissant':
                    target_pos[2] += 150.0
                return target_pos
        return None
    def get_midpoint_of_two_objects(self, object1_name, object2_name):
        self.get_logger().info(f"'{object1_name}'와 '{object2_name}' 두 객체의 중점 위치를 찾습니다.")

        pos1 = self.get_object_position_from_camera(object1_name)
        pos2 = self.get_object_position_from_camera(object2_name)
        robot_posx = get_current_posx()[0]

        if pos1 is None:
            self.get_logger().warn(f"'{object1_name}'을(를) 찾을 수 없어 중점 계산을 할 수 없습니다.")
            return None
        
        if pos2 is None:
            self.get_logger().warn(f"'{object2_name}'을(를) 찾을 수 없어 중점 계산을 할 수 없습니다.")
            return None

        # 두 위치 모두 최소 3개의 구성 요소(x, y, z)를 가지고 있는지 확인합니다.
        if len(pos1) < 3 or len(pos2) < 3:
            self.get_logger().error("객체 위치 데이터에 충분한 3D 정보(x, y, z)가 없습니다.")
            return None

        midpoint_x = (pos1[0] + pos2[0]) / 2
        midpoint_y = (pos1[1] + pos2[1]) / 2
        midpoint_z = (pos1[2] + pos2[2]) / 2

        midpoint_pos = [midpoint_x, midpoint_y, midpoint_z] + robot_posx[3:]
        angle = 90 - math.atan2(pos1[1] - pos2[1], pos1[0] - pos2[0])*180/math.pi

        # 로봇의 방향(robot_posx의 마지막 3개 구성 요소)을 유지해야 한다면,
        # 이를 결합하거나 선택하는 전략이 필요합니다. 여기서는 단순하게
        # 3D 위치만 반환합니다. robot_posx에서 방향을 사용해야 한다면,
        # 다음과 같이 통합할 수 있습니다:
        # midpoint_pos = [midpoint_x, midpoint_y, midpoint_z] + pos1[3:] # 또는 pos2[3:]

        self.get_logger().info(f"'{object1_name}'와 '{object2_name}'의 중점: {midpoint_pos}")
        return midpoint_pos, angle        
    
    def init_robot(self):
        JReady = [0, 0, 90, 0, 90, -90]
        self.get_logger().info("준비 자세로 이동합니다.")
        movej(JReady, vel=VELOCITY, acc=ACC)
        gripper.open_gripper()
        mwait()
    def ready_to_pick_tool(self):
        JPick = [40, 15, 90, 0, 75, 40]
        movej(JPick, vel=VELOCITY, acc=ACC)
        gripper.open_gripper()
        mwait()
    def ready_to_feed_robot(self):
        JFeed = [0, 30, 90, -90, 90, -150]
        self.get_logger().info("먹이기 자세로 이동합니다.")
        movej(JFeed, vel=VELOCITY, acc=ACC)
    
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
    def detecting(self, fc_cond=40):
        task_compliance_ctrl(stx=[500, 500, 500, 100, 100, 100])
        time.sleep(0.1)

        set_desired_force(fd=[0, 0, -fc_cond, 0, 0, 0], dir=[0, 0, 1, 0, 0, 0], mod=DR_FC_MOD_REL)

        while not check_force_condition(DR_AXIS_Z, max=20):
            time.sleep(0.5)
            pass

        release_force()
        time.sleep(0.1)
            
        release_compliance_ctrl()
        time.sleep(0.1)

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
