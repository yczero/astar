import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from nav2_msgs.action import NavigateToPose

import heapq
import numpy as np
from math import atan2, sqrt, sin, pi

class NodeAStar:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0; self.h = 0; self.f = 0
    def __lt__(self, other): return self.f < other.f

class IntegratedNavigation(Node):
    def __init__(self):
        super().__init__('integrated_navigation')

        # --- 핵심 파라미터 조정 (벽 붙기 방지) ---
        self.base_lookahead = 0.4     # 직선 주행 시 주시 거리
        self.min_lookahead = 0.2      # 코너링 시 주시 거리 (낮을수록 코너를 크게 돔)
        self.max_linear_vel = 0.15
        self.max_angular_vel = 0.6    # 회전 속도 제한
        
        # 안전 마진 (로봇 크기 고려)
        self.safety_margin_cells = 6   # 벽으로부터 최소 6칸(30cm) 떨어짐 보장
        self.side_repulsion_dist = 0.35 # 라이다 측면 벽 감지 거리 (이보다 가까우면 반대로 뺌)
        
        self.lidar_data = None
        self.map_data = None
        self.current_pose = None
        self.current_yaw = 0.0
        self.global_path = []
        self.path_index = 0
        self.prev_cmd = Twist()

        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/planned_path', 10)
        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.sub_pose = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self._action_server = ActionServer(self, NavigateToPose, 'navigate_to_pose',
            execute_callback=self.execute_callback, goal_callback=self.goal_callback, cancel_callback=self.cancel_callback)

        self.get_logger().info("Wide-Turn Navigation Node Started.")

    def scan_callback(self, msg):
        self.lidar_data = msg

    def map_callback(self, msg):
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.map_data = np.array(msg.data).reshape((self.map_height, self.map_width))

    def pose_callback(self, msg):
        self.current_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        q = msg.pose.pose.orientation
        self.current_yaw = atan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))

    def goal_callback(self, goal_request): return GoalResponse.ACCEPT
    def cancel_callback(self, goal_handle): return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        if self.map_data is None or self.current_pose is None:
            goal_handle.abort(); return NavigateToPose.Result()

        start_grid = self.world_to_grid(self.current_pose)
        goal_pos = goal_handle.request.pose.pose.position
        goal_grid = self.world_to_grid([goal_pos.x, goal_pos.y])

        # 중앙 주행 유도형 A* 실행
        self.get_logger().info("Planning center-line path...")
        path_grid = self.run_astar_wide(start_grid, goal_grid)
        
        if not path_grid:
            self.get_logger().error("Path not found! Try a more open area.")
            goal_handle.abort(); return NavigateToPose.Result()

        self.global_path = [self.grid_to_world(p) for p in path_grid]
        self.path_index = 0
        self.publish_path_viz()

        rate = self.create_rate(10)
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.stop_robot(); goal_handle.canceled(); return NavigateToPose.Result()

            dist_to_final = sqrt((self.global_path[-1][0]-self.current_pose[0])**2 + (self.global_path[-1][1]-self.current_pose[1])**2)
            if dist_to_final < 0.15: break

            self.wide_control_loop()
            rate.sleep()

        self.stop_robot(); goal_handle.succeed(); return NavigateToPose.Result()

    def wide_control_loop(self):
        """벽에서 멀어지는 조향 보정이 포함된 제어 루프"""
        # 1. 가변 Lookahead (각도가 크면 Lookahead를 줄여서 경로를 엄격히 따름)
        current_lookahead = self.base_lookahead
        
        # 타겟 포인트 찾기
        target_x, target_y = self.global_path[-1]
        for i in range(self.path_index, len(self.global_path)):
            px, py = self.global_path[i]
            dist = sqrt((px - self.current_pose[0])**2 + (py - self.current_pose[1])**2)
            if dist >= current_lookahead:
                target_x, target_y = px, py
                self.path_index = i
                break

        dx = target_x - self.current_pose[0]
        dy = target_y - self.current_pose[1]
        alpha = atan2(dy, dx) - self.current_yaw
        while alpha > pi: alpha -= 2*pi
        while alpha < -pi: alpha += 2*pi

        # 2. 코너링 시 Lookahead 동적 축소 (코너를 더 크게 돌기 위함)
        if abs(alpha) > 0.4:
            current_lookahead = self.min_lookahead

        # 3. 라이다 기반 측면 반발력 (옆구리 충돌 방지 핵심)
        repulsion = 0.0
        if self.lidar_data:
            ranges = np.array(self.lidar_data.ranges)
            ranges = np.where((ranges < 0.05) | np.isinf(ranges), 10.0, ranges)
            
            # 왼쪽/오른쪽 측면 (45도~135도 범위) 확인
            left_side = np.min(ranges[45:135])
            right_side = np.min(ranges[225:315])
            
            if left_side < self.side_repulsion_dist:
                repulsion -= 0.25 * (1.0 - left_side / self.side_repulsion_dist)
            if right_side < self.side_repulsion_dist:
                repulsion += 0.25 * (1.0 - right_side / self.side_repulsion_dist)

        # 4. 속도 명령 생성
        cmd = Twist()
        # 급회전 시 거의 정지 수준으로 감속하여 전복 방지
        slowdown = max(0.1, 1.0 - abs(alpha)/(pi/2))
        cmd.linear.x = self.max_linear_vel * slowdown
        
        # Pure Pursuit + 라이다 반발력
        target_ang = (2.0 * cmd.linear.x * sin(alpha)) / current_lookahead + repulsion
        
        # 각속도 스무딩 (급격한 꺽임 방지)
        ang_diff = target_ang - self.prev_cmd.angular.z
        max_ang_step = 0.15
        cmd.angular.z = self.prev_cmd.angular.z + max(-max_ang_step, min(max_ang_step, ang_diff))
        
        # 제한
        if cmd.angular.z > self.max_angular_vel: cmd.angular.z = self.max_angular_vel
        if cmd.angular.z < -self.max_angular_vel: cmd.angular.z = -self.max_angular_vel

        self.pub_cmd.publish(cmd)
        self.prev_cmd = cmd

    def run_astar_wide(self, start, end):
        """길 한가운데로 다니도록 비용을 계산하는 A*"""
        open_list = []
        heapq.heappush(open_list, (0, NodeAStar(None, start)))
        visited = {}
        
        while open_list:
            _, current = heapq.heappop(open_list)
            if current.position == end:
                path = []
                while current:
                    path.append(current.position); current = current.parent
                return path[::-1]
            
            if current.position in visited and visited[current.position] <= current.g: continue
            visited[current.position] = current.g

            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                nr, nc = current.position[0] + dr, current.position[1] + dc
                if 0 <= nr < self.map_height and 0 <= nc < self.map_width:
                    if self.map_data[nr][nc] == 0:
                        # 벽과의 거리 체크 (가중치 부여)
                        wall_dist_cost = self.get_wall_dist_cost(nr, nc)
                        if wall_dist_cost > 100: continue # 너무 가깝다
                        
                        new_node = NodeAStar(current, (nr, nc))
                        move_cost = 1.4 if abs(dr)+abs(dc)==2 else 1.0
                        new_node.g = current.g + move_cost + wall_dist_cost
                        new_node.h = sqrt((nr - end[0])**2 + (nc - end[1])**2)
                        new_node.f = new_node.g + new_node.h
                        heapq.heappush(open_list, (new_node.f, new_node))
        return None

    def get_wall_dist_cost(self, r, c):
        """벽에 가까울수록 패널티를 줘서 중앙 주행 유도"""
        margin = self.safety_margin_cells
        for d in range(1, margin + 1):
            # 사각형 범위로 벽 탐색
            for dr in range(-d, d + 1):
                for dc in [-d, d]:
                    for nr, nc in [(r+dr, c+dc), (r+dc, c+dr)]:
                        if 0 <= nr < self.map_height and 0 <= nc < self.map_width:
                            if self.map_data[nr][nc] != 0:
                                if d < 3: return 500 # 절대 금지 (15cm)
                                return (margin - d) * 10 # 거리에 따른 가중치
        return 0

    def world_to_grid(self, world):
        return (int((world[1]-self.map_origin[1])/self.map_resolution), int((world[0]-self.map_origin[0])/self.map_resolution))
    def grid_to_world(self, grid):
        return [(grid[1]*self.map_resolution)+self.map_origin[0], (grid[0]*self.map_resolution)+self.map_origin[1]]
    def publish_path_viz(self):
        msg = Path(); msg.header.frame_id = 'map'; msg.header.stamp = self.get_clock().now().to_msg()
        for p in self.global_path:
            ps = PoseStamped(); ps.pose.position.x, ps.pose.position.y = p[0], p[1]
            msg.poses.append(ps)
        self.pub_path.publish(msg)
    def stop_robot(self): self.pub_cmd.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = IntegratedNavigation()
    try: rclpy.spin(node, executor=executor)
    except KeyboardInterrupt: pass
    finally: node.stop_robot(); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()