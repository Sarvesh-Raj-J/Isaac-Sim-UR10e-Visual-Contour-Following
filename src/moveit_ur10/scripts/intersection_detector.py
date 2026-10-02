#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, Point
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import threading
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class RebarDetector(Node):
    def __init__(self):
        super().__init__('rebar_detector')
        self.declare_parameter('rebar_plane_y', 0.8)
        self.rebar_plane_y = self.get_parameter('rebar_plane_y').value
        
        self.bridge = CvBridge()
        self.K = None
        self.latest_image = None
        self.active_3d_target = None
        self.all_rebar_3d_points = [] 
        self.data_lock = threading.Lock() 

        # --- THE FIX: FORCE COMPACT WINDOW SIZE ---
        cv2.namedWindow("Robot Camera View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Robot Camera View", 800, 450) # Scales the display down nicely

        qos_reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.create_subscription(Image, '/rgb', self.rgb_cb, 10)
        self.create_subscription(CameraInfo, '/camera_info', self.info_cb, 10)
        self.create_subscription(Bool, '/detect_intersections', self.trigger_cb, 10)
        self.create_subscription(Point, '/active_rebar_target', self.target_3d_cb, 10)
        
        self.pose_pub = self.create_publisher(PoseArray, '/rebar_intersections', qos_reliable)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.create_timer(0.033, self.viz_timer_cb)
        self.get_logger().info('[DETECTOR] Console Online with Scaled Window.')

    def target_3d_cb(self, msg):
        with self.data_lock:
            self.active_3d_target = np.array([msg.x, msg.y, msg.z])

    def info_cb(self, msg):
        if self.K is None: self.K = np.array(msg.k).reshape(3, 3)

    def rgb_cb(self, msg):
        with self.data_lock: self.latest_image = msg

    def viz_timer_cb(self):
        with self.data_lock:
            img_msg, target_3d, points_3d = self.latest_image, self.active_3d_target, self.all_rebar_3d_points
            
        if img_msg is None or self.K is None: return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            vis_img = cv2.convertScaleAbs(cv_img, alpha=1.2, beta=30)

            try:
                tf = self.tf_buffer.lookup_transform('ZED_X', 'base_link', rclpy.time.Time())
                q, t = tf.transform.rotation, tf.transform.translation
                rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                trans = np.array([t.x, t.y, t.z])

                # 1. Project Map Points (Red)
                for pt in points_3d:
                    u, v = self.project_3d_to_2d(pt, rot, trans)
                    if u is not None:
                        cv2.circle(vis_img, (u, v), 6, (0, 0, 255), -1)

                # 2. Project Active Target (Yellow)
                if target_3d is not None:
                    u, v = self.project_3d_to_2d(target_3d, rot, trans)
                    if u is not None:
                        cv2.circle(vis_img, (u, v), 18, (0, 255, 255), 3)
                        cv2.putText(vis_img, "TARGET", (u+25, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            except: pass

            cv2.putText(vis_img, "REAL-TIME AR TRACKING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Robot Camera View", vis_img)
            cv2.waitKey(1)
        except Exception as e: self.get_logger().error(f'GUI Error: {e}')

    def project_3d_to_2d(self, pt_3d, rot, trans):
        pt_cam = rot @ pt_3d + trans
        if pt_cam[2] <= 0.1: return None, None
        u = int((self.K[0,0] * pt_cam[0] / pt_cam[2]) + self.K[0,2])
        v = int((self.K[1,1] * pt_cam[1] / pt_cam[2]) + self.K[1,2])
        if 0 <= u < 1280 and 0 <= v < 720: 
            return u, v
        return None, None

    def trigger_cb(self, msg):
        if msg.data: threading.Thread(target=self.detect_and_publish).start()

    def detect_and_publish(self):
        with self.data_lock:
            if self.latest_image is None or self.K is None: return
            curr_msg = self.latest_image

        try:
            cv_img = self.bridge.imgmsg_to_2026_cv2(curr_msg, 'bgr8') if hasattr(self.bridge, 'imgmsg_to_2026_cv2') else self.bridge.imgmsg_to_cv2(curr_msg, 'bgr8')
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=80, maxLineGap=30)
            
            pixel_pts = self.detect_intersections(lines)
            
            temp_3d_points = []
            poses = []
            for (u, v) in pixel_pts:
                pt_3d = self.project_pixels_to_3d_plane(u, v)
                if pt_3d is not None:
                    temp_3d_points.append(pt_3d)
                    p = Pose()
                    p.position.x, p.position.y, p.position.z = map(float, pt_3d)
                    p.orientation.w = 1.0
                    poses.append(p)

            with self.data_lock:
                self.all_rebar_3d_points = temp_3d_points

            pa = PoseArray()
            pa.header.frame_id, pa.header.stamp = 'base_link', self.get_clock().now().to_msg()
            pa.poses = poses
            self.pose_pub.publish(pa)
            self.get_logger().info(f'[DETECTOR] Mapped {len(poses)} intersections to 3D.')
        except Exception as e: self.get_logger().error(f'Logic error: {e}')

    def project_pixels_to_3d_plane(self, u, v):
        try:
            tf = self.tf_buffer.lookup_transform('base_link', 'ZED_X', rclpy.time.Time())
            ray_cam = np.array([(u - self.K[0,2])/self.K[0,0], (v - self.K[1,2])/self.K[1,1], -1.0])
            ray_cam /= np.linalg.norm(ray_cam)
            rot = R.from_quat([tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]).as_matrix()
            ray_base = rot @ ray_cam
            origin = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
            s = (self.rebar_plane_y - origin[1]) / ray_base[1]
            return origin + s * ray_base if s > 0 else None
        except: return None

    def detect_intersections(self, lines):
        if lines is None: return []
        h, v = [], []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            if abs(angle) < 15 or abs(angle) > 165: h.append(l[0])
            elif abs(abs(angle)-90) < 15: v.append(l[0])
        pts = []
        for hl in h:
            for vl in v:
                res = self.line_intersect(hl, vl)
                if res: pts.append(res)
        return self.cluster_points(pts, radius=50)

    @staticmethod
    def line_intersect(l1, l2):
        x1, y1, x2, y2 = l1; x3, y3, x4, y4 = l2
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-6: return None
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        return (int(x1+t*(x2-x1)), int(y1+t*(y2-y1)))

    @staticmethod
    def cluster_points(points, radius):
        if not points: return []
        used, clustered = [False]*len(points), []
        for i, p in enumerate(points):
            if used[i]: continue
            c = [p]; used[i] = True
            for j in range(i+1, len(points)):
                if not used[j] and np.linalg.norm(np.array(p)-np.array(points[j])) < radius:
                    c.append(points[j]); used[j] = True
            clustered.append((int(np.mean([x[0] for x in c])), int(np.mean([x[1] for x in c]))))
        return clustered

def main():
    rclpy.init(); node = RebarDetector()
    try: rclpy.spin(node)
    except: pass
    finally: cv2.destroyAllWindows(); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
