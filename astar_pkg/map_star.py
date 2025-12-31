import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.action import NavigateToPose  # Nav2 Goal 버튼이 사용하는 액션

import heapq
import numpy as np
from math import atan2, sqrt, sin, pi

class NodeAStar:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
    def __eq__(self, other): return self.position == other.position
    def __lt__(self, other): return self.f < other.f

class IntegratedNavigation(Node):
    def __init__(self):
        super().__init__('integrated_navigation')

        # 제어 파라미터
        self.lookahead_dist = 0.4
        self.linear_vel = 0.2
        self.stop_tolerance = 0.15
        self.safety_margin = 3
        
        self.map_data = None
        self.map_origin = [0.0, 0.0]
        self.current_pose = None
        self.current_yaw = 0.0
        self.global_path = []
        self.path_index = 0

        # QoS 설정 (Map 전용)
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publisher / Subscriber
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/planned_path', 10)
        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.sub_pose = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)

        # [핵심] NavigateToPose 액션 서버 생성
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.get_logger().info("Action Server Ready. Use 'Nav2 Goal' in RViz!")

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

    # 액션 목표 수락 여부 결정
    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    # 액션 취소 처리
    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    # 실제 주행 로직 (Nav2 Goal 클릭 시 실행됨)
    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        
        if self.map_data is None or self.current_pose is None:
            self.get_logger().error('Map or Pose not available!')
            goal_handle.abort()
            return NavigateToPose.Result()

        # 1. 경로 탐색 (A*)
        target = goal_handle.request.pose.pose.position
        goal_grid = self.world_to_grid([target.x, target.y])
        start_grid = self.world_to_grid(self.current_pose)

        self.get_logger().info("Calculating A* Path...")
        path_grid = self.run_astar(start_grid, goal_grid)

        if not path_grid:
            self.get_logger().warn("Path not found!")
            goal_handle.abort()
            return NavigateToPose.Result()

        self.global_path = [self.grid_to_world(p) for p in path_grid]
        self.path_index = 0
        self.publish_path_viz()

        # 2. 제어 루프 (도착할 때까지 반복)
        feedback_msg = NavigateToPose.Feedback()
        result = NavigateToPose.Result()

        rate = self.create_rate(10) # 10Hz
        
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.stop_robot()
                goal_handle.canceled()
                return result

            # 도착 확인
            final_goal = self.global_path[-1]
            dist_to_final = sqrt((final_goal[0]-self.current_pose[0])**2 + (final_goal[1]-self.current_pose[1])**2)
            
            if dist_to_final < self.stop_tolerance:
                break

            # Pure Pursuit 제어
            self.control_step()
            
            # 피드백 전송 (선택 사항)
            # feedback_msg.distance_remaining = dist_to_final
            # goal_handle.publish_feedback(feedback_msg)

            rate.sleep()

        self.stop_robot()
        goal_handle.succeed()
        self.get_logger().info("Goal Reached Successfully!")
        return result

    def control_step(self):
        # 현실적인 타겟 갱신 (Lookahead)
        target_x, target_y = self.global_path[-1]
        for i in range(self.path_index, len(self.global_path)):
            px, py = self.global_path[i]
            dist = sqrt((px - self.current_pose[0])**2 + (py - self.current_pose[1])**2)
            if dist >= self.lookahead_dist:
                target_x, target_y = px, py
                self.path_index = i
                break

        dx = target_x - self.current_pose[0]
        dy = target_y - self.current_pose[1]
        alpha = atan2(dy, dx) - self.current_yaw
        
        while alpha > pi: alpha -= 2*pi
        while alpha < -pi: alpha += 2*pi
        
        cmd = Twist()
        cmd.linear.x = self.linear_vel
        cmd.angular.z = (2.0 * self.linear_vel * sin(alpha)) / self.lookahead_dist
        
        # 속도 제한
        if cmd.angular.z > 1.0: cmd.angular.z = 1.0
        if cmd.angular.z < -1.0: cmd.angular.z = -1.0
        
        self.pub_cmd.publish(cmd)

    def check_safety(self, y, x):
        margin = self.safety_margin
        for r in range(y - margin, y + margin + 1):
            for c in range(x - margin, x + margin + 1):
                if 0 <= r < self.map_height and 0 <= c < self.map_width:
                    if self.map_data[r][c] > 50 or self.map_data[r][c] == -1:
                        return False
        return True

    def run_astar(self, start, end):
        if not (0 <= start[0] < self.map_height and 0 <= start[1] < self.map_width): return None
        start_node = NodeAStar(None, start)
        end_node = NodeAStar(None, end)
        open_list = []
        heapq.heappush(open_list, start_node)
        visited = set()
        moves = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]

        while open_list:
            current_node = heapq.heappop(open_list)
            if current_node.position in visited: continue
            visited.add(current_node.position)

            if sqrt((current_node.position[0]-end[0])**2 + (current_node.position[1]-end[1])**2) < 1.5:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]

            for move in moves:
                ny, nx = current_node.position[0] + move[0], current_node.position[1] + move[1]
                if 0 <= ny < self.map_height and 0 <= nx < self.map_width:
                    if self.map_data[ny][nx] <= 50 and self.check_safety(ny, nx):
                        new_node = NodeAStar(current_node, (ny, nx))
                        new_node.g = current_node.g + 1
                        new_node.h = sqrt((ny - end[0])**2 + (nx - end[1])**2)
                        new_node.f = new_node.g + new_node.h
                        heapq.heappush(open_list, new_node)
        return None

    def world_to_grid(self, world):
        return (int((world[1]-self.map_origin[1])/self.map_resolution), int((world[0]-self.map_origin[0])/self.map_resolution))

    def grid_to_world(self, grid):
        return [(grid[1]*self.map_resolution)+self.map_origin[0], (grid[0]*self.map_resolution)+self.map_origin[1]]

    def publish_path_viz(self):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for p in self.global_path:
            ps = PoseStamped()
            ps.pose.position.x, ps.pose.position.y = p[0], p[1]
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def stop_robot(self):
        self.pub_cmd.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    # Action Server는 멀티스레딩이 권장됩니다.
    node = IntegratedNavigation()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()