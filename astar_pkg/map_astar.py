import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan, Image

from rclpy.action import ActionServer
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from math import atan2, sqrt, pi
import heapq
import numpy as np
import cv2

from cv_bridge import CvBridge
from ultralytics import YOLO


# ================= A* Node =================
class NodeAStar:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f


# ================= Main Node =================
class IntegratedNavigation(Node):

    def __init__(self):
        super().__init__('integrated_navigation')

        # =======================
        # 🔧 튜닝 파라미터
        # =======================
        self.lookahead_dist = 0.6
        self.linear_vel = 0.18
        self.min_linear = 0.05

        self.max_angular = 0.8
        self.front_safe_dist = 0.45
        self.side_safe_dist = 0.35

        self.stop_tolerance = 0.2
        self.robot_radius = 0.25
        # =======================

        # ===== Lidar =====
        self.front_min = 999.0
        self.left_min = 999.0
        self.right_min = 999.0

        # ===== Map =====
        self.map_data = None
        self.inflated_map = None
        self.map_resolution = 0.05
        self.map_origin = [0.0, 0.0]
        self.map_width = 0
        self.map_height = 0
        self.inflation_radius = 0

        # ===== Robot State =====
        self.current_pose = None
        self.current_yaw = 0.0

        # ===== Path =====
        self.global_path = []
        self.path_index = 0

        # ===== YOLO Emergency Stop =====
        self.bridge = CvBridge()
        self.emergency_stop = False
        self.yolo_model = YOLO('/home/zero/ros2_ws/yolov8n.pt')  # 컵 포함 COCO 모델

        # ===== QoS =====
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        pose_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # ===== Pub / Sub =====
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/planned_path', 10)

        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, pose_qos)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        # ===== Nav2 Action =====
        self.action_server = ActionServer(
            self,
            NavigateToPose,
            '/navigate_to_pose',
            self.execute_nav_goal
        )

        self.timer = self.create_timer(0.1, self.control_loop)

    # ================= LIDAR =================

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment

        def get_min(start_deg, end_deg):
            s = int((start_deg*pi/180 - angle_min) / angle_inc)
            e = int((end_deg*pi/180 - angle_min) / angle_inc)
            r = ranges[s:e]
            r = r[np.isfinite(r)]
            return np.min(r) if len(r) > 0 else 999.0

        self.front_min = get_min(-25, 25)
        self.left_min = get_min(30, 90)
        self.right_min = get_min(-90, -30)

    # ================= YOLO (컵 비상정지) =================

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.yolo_model(frame, conf=0.5, verbose=False)

        cup_detected = False

        for r in results:
            if r.boxes is None:
                continue

            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                class_name = self.yolo_model.names[int(cls)]

                # ✅ 컵일 때만 비상정지
                if class_name == 'cup':
                    cup_detected = True

                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, 'EMERGENCY STOP (CUP)',
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)

        self.emergency_stop = cup_detected

        cv2.imshow('YOLO Emergency Detection', frame)
        cv2.waitKey(1)

    # ================= MAP =================

    def map_callback(self, msg):
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = [
            msg.info.origin.position.x,
            msg.info.origin.position.y
        ]

        self.map_data = np.array(msg.data).reshape(
            (self.map_height, self.map_width)
        )

        self.inflation_radius = int(self.robot_radius / self.map_resolution)
        self.inflated_map = self.inflate_map()

    def inflate_map(self):
        inflated = self.map_data.copy()
        for y in range(self.map_height):
            for x in range(self.map_width):
                if self.map_data[y][x] == 100:
                    for dy in range(-self.inflation_radius, self.inflation_radius + 1):
                        for dx in range(-self.inflation_radius, self.inflation_radius + 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.map_height and 0 <= nx < self.map_width:
                                inflated[ny][nx] = 100
        return inflated

    # ================= POSE =================

    def pose_callback(self, msg):
        self.current_pose = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ]
        q = msg.pose.pose.orientation
        self.current_yaw = atan2(
            2.0 * (q.w*q.z + q.x*q.y),
            1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        )

    # ================= ACTION =================

    def execute_nav_goal(self, goal_handle):
        if self.map_data is None or self.current_pose is None:
            goal_handle.abort()
            return NavigateToPose.Result()

        goal = goal_handle.request.pose.pose.position
        start = self.world_to_grid(self.current_pose)
        goal_grid = self.world_to_grid([goal.x, goal.y])

        path = self.run_astar(start, goal_grid)
        if not path:
            goal_handle.abort()
            return NavigateToPose.Result()

        self.global_path = [self.grid_to_world(p) for p in path]
        self.path_index = 0
        self.publish_path_viz()

        while rclpy.ok() and self.global_path:
            rclpy.spin_once(self, timeout_sec=0.1)

        goal_handle.succeed()
        return NavigateToPose.Result()

    # ================= A* =================

    def run_astar(self, start, end):
        open_list = []
        closed = set()

        heapq.heappush(open_list, NodeAStar(None, start))
        moves = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

        while open_list:
            current = heapq.heappop(open_list)
            if current.position in closed:
                continue
            closed.add(current.position)

            if current.position == end:
                path = []
                while current:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]

            for dy, dx in moves:
                ny = current.position[0] + dy
                nx = current.position[1] + dx
                if not self.valid_grid((ny, nx)):
                    continue
                if self.inflated_map[ny][nx] == 100:
                    continue

                child = NodeAStar(current, (ny, nx))
                child.g = current.g + 1
                child.h = sqrt((ny-end[0])**2 + (nx-end[1])**2)
                child.f = child.g + child.h
                heapq.heappush(open_list, child)
        return None

    # ================= CONTROL =================

    def control_loop(self):

        # 🚨 컵 감지 시 최우선 비상정지
        if self.emergency_stop:
            self.stop_robot()
            return

        if self.current_pose is None or not self.global_path:
            return

        gx, gy = self.global_path[-1]
        if sqrt((gx - self.current_pose[0])**2 +
                (gy - self.current_pose[1])**2) < self.stop_tolerance:
            self.stop_robot()
            self.global_path = []
            return

        target = None
        for i in range(self.path_index, len(self.global_path)):
            px, py = self.global_path[i]
            if sqrt((px - self.current_pose[0])**2 +
                    (py - self.current_pose[1])**2) >= self.lookahead_dist:
                target = (px, py)
                self.path_index = i
                break

        if target is None:
            target = self.global_path[-1]

        angle_target = atan2(
            target[1] - self.current_pose[1],
            target[0] - self.current_pose[0]
        )

        angle_error = (angle_target - self.current_yaw + pi) % (2*pi) - pi
        angular = 2.0 * angle_error
        angular = max(-self.max_angular, min(self.max_angular, angular))

        if self.front_min < self.front_safe_dist:
            angular = 0.6 if self.left_min > self.right_min else -0.6
            linear = 0.0
        else:
            linear = self.linear_vel * max(0.3, 1.0 - abs(angular))

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.pub_cmd.publish(cmd)

    # ================= UTILS =================

    def valid_grid(self, grid):
        y, x = grid
        return 0 <= y < self.map_height and 0 <= x < self.map_width

    def world_to_grid(self, w):
        return (
            int((w[1] - self.map_origin[1]) / self.map_resolution),
            int((w[0] - self.map_origin[0]) / self.map_resolution)
        )

    def grid_to_world(self, g):
        return [
            g[1] * self.map_resolution + self.map_origin[0],
            g[0] * self.map_resolution + self.map_origin[1]
        ]

    def publish_path_viz(self):
        msg = Path()
        msg.header.frame_id = 'map'
        for p in self.global_path:
            ps = PoseStamped()
            ps.pose.position.x = p[0]
            ps.pose.position.y = p[1]
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def stop_robot(self):
        self.pub_cmd.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedNavigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
