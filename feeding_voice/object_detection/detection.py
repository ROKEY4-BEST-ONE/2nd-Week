import numpy as np
import cv2
import rclpy, time
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse

from ament_index_python.packages import get_package_share_directory
from hh_od_msg.action import TrackLips
from hh_od_msg.srv import SrvDepthPosition, SrvRiceRichPosition, SrvCheckStop
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

        self._action_server = ActionServer(
            self,
            TrackLips,
            'track_lips',
            execute_callback=self.execute_callback,
            cancel_callback=self.cancel_callback
        )
        self.create_service(
            SrvRiceRichPosition,
            'get_rice_rich_position',
            self.handle_rice_depth
        )
        self.create_service(
            SrvDepthPosition,
            'get_3d_position',
            self.handle_get_depth
        )
        self.get_logger().info("ObjectDetectionNode initialized.")

    def _load_model(self, name, target):
        # target을 탐지하기 위한 모델을 load합니다.
        if target == 'pikachu' or target == 'Bulbasaur':
            category = 'pokemon'
        elif target == 'pororo' or target == 'loopy':
            category = 'pororo'
        elif target == 'Broccoli' or target == 'Croissant' or target == 'Macaron':
            category = 'food'
        else:
            category = target
        if name.lower() == 'yolo':
            return YoloModel(category)
        raise ValueError(f"Unsupported model: {name}")

    async def execute_callback(self, goal_handle):
        # 입술의 depth position을 액션 피드백으로 전송하는 함수
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

            check_stop_client = self.create_client(SrvCheckStop, "/check_stop")
            while not check_stop_client.wait_for_service(timeout_sec=3.0):
                self.get_logger().error("check_stop_pos 서비스를 찾을 수 없습니다.")
            check_stop_request = SrvCheckStop.Request()
            check_stop_request.coords = [float(x) for x in coords]
            check_stop_future = check_stop_client.call_async(check_stop_request)
            rclpy.spin_until_future_complete(self, check_stop_future)
            if check_stop_future.result():
                if check_stop_future.result().stop:
                    break

            # time.sleep(1)

        goal_handle.succeed()
        success = False
        result = TrackLips.Result()
        result.success = success
        self.get_logger().info(f"Result: success={success}")
        return result

    def cancel_callback(self, goal_handle):
        # 액션을 취소하는 함수 (사용하지는 않음)
        self.get_logger().info('Goal 취소 요청 수신.')
        return CancelResponse.ACCEPT
    
    def handle_rice_depth(self, request, response):
        # 클라이언트 요청 처리 → 쌀 많은 구역의 카메라 좌표 반환
        self.model = self._load_model('yolo', 'rice')
        self.get_logger().info("Received request: 'rice'")
        coords = self._compute_rice_region_position()
        response.depth_position = [float(x) for x in coords]
        return response

    def _compute_rice_region_position(self):
        """
        YOLO segmentation으로 쌀 탐지 → 가장 큰 덩어리 중 스코어 높은 마스크 선택 →
        마스크 내부에서 밀도 높은 중심점 → Depth 기반 카메라 좌표 반환
        """
        # 1. YOLO로 가장 좋은 마스크 찾기
        box, mask, score = self.model.get_best_masked_detection(self.img_node, 'rice')
        if mask is None:
            self.get_logger().warn("No rice detected with segmentation.")
            return 0.0, 0.0, 0.0

        # 2. 마스크 내 밀도 기반 중심점 계산
        # (단순 centroid도 가능하지만 숟가락 크기 고려 시 슬라이딩 윈도우 방식 추천)
        spoon_size = 70  # 픽셀 단위, 실제 숟가락 직경에 맞춰 조정
        kernel = np.ones((spoon_size, spoon_size), np.uint8)
        density_map = cv2.filter2D(mask.astype(np.uint8), -1, kernel)
        _, _, _, maxLoc = cv2.minMaxLoc(density_map)
        cx, cy = maxLoc  # (x, y)

        # 3. Depth 값 추출
        cz = self._get_depth(cx, cy)
        if cz is None:
            self.get_logger().warn("Depth not available at selected point.")
            return 0.0, 0.0, 0.0

        # 4. 카메라 좌표 변환
        return self._pixel_to_camera_coords(cx, cy, cz, 'rice')

    def handle_get_depth(self, request, response):
        # 클라이언트 요청을 처리해 depth position을 반환합니다.
        self.model = self._load_model('yolo', request.target)
        self.get_logger().info(f"Received request: {request}")
        coords = self._compute_position(request.target)
        response.depth_position = [float(x) for x in coords]
        return response

    def _compute_position(self, target):
        # 이미지를 처리해 객체의 카메라 좌표를 계산합니다.
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
        # 픽셀 좌표의 depth 값을 안전하게 읽어옵니다.
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
        # 픽셀 좌표와 intrinsics를 이용해 카메라 좌표계로 변환합니다.
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
