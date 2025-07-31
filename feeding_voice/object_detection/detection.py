import numpy as np
import rclpy, time, math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse

from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float64MultiArray
from hh_od_msg.action import TrackLips
from hh_od_msg.srv import SrvDepthPosition
from object_detection.realsense import ImgNode
from object_detection.yolo import YoloModel


PACKAGE_NAME = 'feeding_voice'
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)


class ObjectDetectionNode(Node):
    def __init__(self, model_name = 'yolo'):
        super().__init__('object_detection_node')
        self.img_node = ImgNode()
        self.lips_model = self._load_model(model_name, 'lips')
        self.intrinsics = self._wait_for_valid_data(
            self.img_node.get_camera_intrinsic, "camera intrinsics"
        )
        while self.intrinsics is None:
            time.sleep(0.5)
            self.intrinsics = self._wait_for_valid_data(
                self.img_node.get_camera_intrinsic, "camera intrinsics"
            )

        # ✅ 최신 로봇 좌표 저장 변수
        self.current_robot_pos = [100.0, 0.0, 100.0]

        # ✅ 로봇 좌표 Subscriber
        self.create_subscription(Float64MultiArray, '/robot_position', self.robot_pos_callback, 10)

        self._action_server = ActionServer(
            self,
            TrackLips,
            'track_lips',
            execute_callback=self.execute_callback,
            cancel_callback=self.cancel_callback
        )
        self.create_service(
            SrvDepthPosition,
            'get_3d_position',
            self.handle_get_depth
        )
        self.get_logger().info("ObjectDetectionNode initialized.")

    def _load_model(self, name, target):
        """모델 이름에 따라 인스턴스를 반환합니다."""
        if target == 'pikachu' or target == 'Bulbasaur':
            category = 'pokemon'
        else:
            category = target
        if name.lower() == 'yolo':
            return YoloModel(category)
        raise ValueError(f"Unsupported model: {name}")

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Goal 수신. 입술 추적 시작.')
        feedback_msg = TrackLips.Feedback()
        success = False

        self.model = self._load_model('yolo', 'lips')

        # 특정 조건 만족할 때까지 반복
        while rclpy.ok():
            rclpy.spin_once(self.img_node, timeout_sec=0.1)

            # YOLO로 좌표 계산
            coords = self._compute_position('lips')
            if coords == (0.0, 0.0, 0.0):
                self.get_logger().warn("입술 탐지 실패.")
                continue

            feedback_msg.depth_position = [float(x) for x in coords]
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f"피드백 전송: {feedback_msg.depth_position}")

            # 로봇 위치와 비교하는 로직 (가정: ROS 토픽 또는 서비스에서 로봇 현재 위치 가져옴)
            robot_pos = self._get_robot_position()  # [x, y, z]
            dist = math.dist(robot_pos, coords)
            if dist < 50.0:  # threshold (단위는 mm 가정)
                success = True
                break

            time.sleep(10)

        goal_handle.succeed()
        result = TrackLips.Result()
        result.success = success
        self.get_logger().info(f"Result: success={success}")
        return result
    
    def robot_pos_callback(self, msg):
        """로봇 제어 노드에서 퍼블리시하는 좌표를 최신 값으로 저장"""
        self.current_robot_pos = list(msg.data)

    def _get_robot_position(self):
        """토픽에서 받은 최신 로봇 좌표 반환"""
        return self.current_robot_pos

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Goal 취소 요청 수신.')
        return CancelResponse.ACCEPT

    def handle_get_depth(self, request, response):
        """클라이언트 요청을 처리해 3D 좌표를 반환합니다."""
        self.model = self._load_model('yolo', request.target)
        self.get_logger().info(f"Received request: {request}")
        coords = self._compute_position(request.target)
        response.depth_position = [float(x) for x in coords]
        return response

    def _compute_position(self, target):
        """이미지를 처리해 객체의 카메라 좌표를 계산합니다."""
        rclpy.spin_once(self.img_node)

        box, score = self.model.get_best_detection(self.img_node, target)
        if box is None or score is None:
            self.get_logger().warn("No detection found.")
            return 0.0, 0.0, 0.0
        
        self.get_logger().info(f"Detection: box={box}, score={score}")
        cx, cy = map(int, [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        cz = self._get_depth(cx, cy)
        if cz is None:
            self.get_logger().warn("Depth out of range.")
            return 0.0, 0.0, 0.0

        return self._pixel_to_camera_coords(cx, cy, cz, target)

    def _get_depth(self, x, y):
        """픽셀 좌표의 depth 값을 안전하게 읽어옵니다."""
        frame = self._wait_for_valid_data(self.img_node.get_depth_frame, "depth frame")
        try:
            return frame[y, x]
        except IndexError:
            self.get_logger().warn(f"Coordinates ({x},{y}) out of range.")
            return None

    def _wait_for_valid_data(self, getter, description):
        """getter 함수가 유효한 데이터를 반환할 때까지 spin 하며 재시도합니다."""
        data = getter()
        while data is None or (isinstance(data, np.ndarray) and not data.any()):
            rclpy.spin_once(self.img_node)
            self.get_logger().info(f"Retry getting {description}.")
            data = getter()
        return data

    def _pixel_to_camera_coords(self, x, y, z, target):
        """픽셀 좌표와 intrinsics를 이용해 카메라 좌표계로 변환합니다."""
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        ppx = self.intrinsics['ppx']
        ppy = self.intrinsics['ppy']
        if target == 'apple':
            z_offset = 60
        elif target == 'lips':
            z_offset = -160
        else:
            z_offset = 0
        return (
            (x - ppx) * z / fx,
            (y - ppy) * z / fy,
            z + z_offset
        )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
