import os
import time
import sys
import math
import threading
import queue
from scipy.spatial.transform import Rotation
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import DR_init
from hh_od_msg.action import TrackLips
from hh_od_msg.srv import SrvDepthPosition
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
    from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, amovel, drl_script_stop, DR_MV_MOD_REL, DR_SSTOP
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
        self.track_lips_action_client = ActionClient(self, TrackLips, 'track_lips')
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

        self.move_queue = queue.Queue(maxsize=1)  # 큐 사이즈 = 1 → 항상 최신 좌표 유지
        self.stop_thread = False
        self.pause_thread = True  # 초기에는 대기 상태

        # 워커 스레드 시작
        self.worker_thread = threading.Thread(target=self.move_worker, daemon=True)
        self.worker_thread.start()

        self.current_robot_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 스레드: 로봇 좌표 갱신
        self.update_thread = threading.Thread(target=self.update_robot_position, daemon=True)
        self.update_thread.start()
        
        self.command_queue = queue.Queue()

        self.create_timer(0.1, self.process_commands)  # 10Hz
    
    def process_commands(self):
        if not self.command_queue.empty():
            target_pos = self.command_queue.get()
            self.get_logger().info(f"[ROS] movel 실행: {target_pos}")
            try:
                drl_script_stop(DR_SSTOP)
                movel(target_pos, vel=20, acc=20)  # ✅ Executor 안에서 실행 → 안전
            except Exception as e:
                self.get_logger().error(f"movel 실행 오류: {e}")


    def update_robot_position(self):
        """ROS 메인 Executor가 아닌 별도 스레드에서 get_current_posx()를 직접 호출하지 않는다.
        대신 rclpy.spin()이 돌고 있으므로 Timer로 갱신."""
        def update():
            try:
                pos = get_current_posx()[0]
                self.current_robot_pos = pos
            except Exception as e:
                self.get_logger().error(f"좌표 갱신 실패: {e}")
        # ROS Timer 사용 (메인 스레드에서 실행)
        self.create_timer(0.1, update)  # 10Hz 주기

    def feedback_callback(self, feedback_msg):
        depth_pos = feedback_msg.feedback.depth_position
        if len(depth_pos) >= 3:
            # 큐가 꽉 차면 오래된 값 버리고 새 값 넣음
            if not self.move_queue.empty():
                try:
                    self.move_queue.get_nowait()  # 오래된 좌표 버림
                except queue.Empty:
                    pass
            self.move_queue.put(depth_pos)

    def move_worker(self):
        while not self.stop_thread:
            if self.pause_thread:
                time.sleep(0.1)
                continue
            try:
                depth_pos = self.move_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            current_pos = self.current_robot_pos
            dist = math.dist(current_pos[:3], depth_pos[:3])
            if dist < 5.0:
                continue

            target_pos = list(depth_pos[:3]) + current_pos[3:]
            self.get_logger().info(f"[워커] 이동 요청 추가: {target_pos}")
            self.command_queue.put(target_pos)  # ✅ ROS API 대신 큐에 넣기


    def start_tracking(self):
        self.pause_thread = False  # 워커 활성화
        self.get_logger().info("[워커] 활성화")

    def stop_tracking(self):
        self.pause_thread = True  # 워커 비활성화
        self.get_logger().info("[워커] 비활성화")

    def shutdown_worker(self):
        self.stop_thread = True
        self.worker_thread.join()

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
                # feed_destination = 'lips'
            except (ValueError, IndexError):
                self.get_logger().error(f"잘못된 형식의 응답입니다: '{response_message}'. '도구 / 목적지' 형식이 필요합니다.")
                return
            # # 1. Pick 위치 (카메라로 숟가락 or 포크 찾기)
            # if food_to_eat == 'rice':
            #     self.ready_to_pick_tool()
            #     tool_name = 'spoon'
            #     pick_tool_pos, pick_tool_angle = self.get_midpoint_of_two_objects('Bulbasaur', 'pikachu')
            #     if pick_tool_pos is None:
            #         self.get_logger().error(f"'{tool_name}'을(를) 찾지 못해 작업을 중단합니다.")
            #         return
            #     amovel(pick_tool_pos, vel=VELOCITY, acc=ACC)
            #     movej([0, 0, 0, 0, 0, pick_tool_angle], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
            #     mwait()
            #     movel([0, 0, -70, 0, 0, 0], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
            #     gripper.close_gripper()
            #     while gripper.get_status()[0]:
            #         time.sleep(0.1)
            #     movel([0, 0, 110, 0, 0, 0], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
            #     movej([0, 0, 90, 0, 90, -90], vel=VELOCITY, acc=ACC)
            # 2. Pick 위치 (카메라로 '사과' 찾기)
            self.get_logger().info(f"--- Pick 대상 '{food_to_eat}' 위치 찾는 중 ---")
            pick_pos = self.get_object_position_from_camera(food_to_eat)
            if pick_pos is None:
                self.get_logger().error(f"'{food_to_eat}'을(를) 찾지 못해 작업을 중단합니다.")
                return
            self.pick_food(pick_pos)
            # 3. 먹이기 (카메라로 '입술' 찾기)
            self.ready_to_feed_robot()
            self.get_logger().info("입술 추적 시작...")

            self.start_tracking()  # 워커 활성화

            self.track_lips_action_client.wait_for_server()
            goal_msg = TrackLips.Goal()
            send_goal_future = self.track_lips_action_client.send_goal_async(
                goal_msg,
                feedback_callback=self.feedback_callback
            )
            rclpy.spin_until_future_complete(self, send_goal_future)

            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error("TrackLips Goal이 거부되었습니다.")
                self.stop_tracking()
                return

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result().result

            self.stop_tracking()  # 추적 비활성화

            if result.success:
                self.get_logger().info("✅ 입술 근접 완료! init_robot() 수행")
            else:
                self.get_logger().warn("입술 추적 실패. init_robot() 수행")

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
        JFeed = [0, 0, 90, -90, 90, -180]
        self.get_logger().info("먹이기 자세로 이동합니다.")
        movej(JFeed, vel=VELOCITY, acc=ACC)
    def pick_tool(self, pick_pos):
        pass
    def pick_food(self, pick_pos):
        self.get_logger().info(f"Pick 위치로 이동: {pick_pos[:3]}")
        movel(pick_pos, vel=VELOCITY, acc=ACC)
        mwait()
        gripper.close_gripper()
        while gripper.get_status()[0]:
            time.sleep(0.1)
        self.get_logger().info("Pick 완료.")
        JReady = [0, 0, 90, 0, 90, 0]
        movej(JReady, vel=VELOCITY, acc=ACC)
        # :둥근_압핀: 안전을 위해 입술 근처 Z축에 오프셋을 둘 수 있습니다.
        # place_pos_safe = place_pos[:]
        # place_pos_safe[2] += 50.0 # Z축으로 5cm 위까지만 접근
        # movel(place_pos_safe, vel=VELOCITY, acc=ACC)
        # self.get_logger().info("Place 완료. 그리퍼를 엽니다.")
        # gripper.open_gripper()
        # while gripper.get_status()[0]:
        #     time.sleep(0.1)
    def feed_food(self, feed_pos):
        self.get_logger().info(f"Pick 위치로 이동: {feed_pos[:3]}")
        movel(feed_pos, vel=VELOCITY, acc=ACC)
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
