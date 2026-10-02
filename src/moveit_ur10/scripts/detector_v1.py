#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, Point
from std_msgs.msg import Bool, Int32
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

        self.bridge              = CvBridge()
        self.K                   = None
        self.latest_image        = None
        self.active_index        = -1
        self.all_rebar_3d_points = []   # base_link 3D — for supervisor only
        self.data_lock           = threading.Lock()

        # ── The one thing stored at publish-time:
        #    list of (u, v) pixel centroids from the SCAN frame.
        #    Used only to match active_index → pixel for the green highlight.
        self.published_pixels = []   # index-aligned with /rebar_intersections poses

        cv2.namedWindow("Robot Camera View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Robot Camera View", 800, 450)

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Image,      '/rgb',                  self.rgb_cb,       10)
        self.create_subscription(CameraInfo, '/camera_info',          self.info_cb,      10)
        self.create_subscription(Bool,       '/detect_intersections', self.trigger_cb,   10)
        self.create_subscription(Point,      '/active_rebar_target',  self.target_3d_cb, 10)
        self.create_subscription(Int32,      '/active_rebar_index',   self.index_cb,     10)

        self.pose_pub    = self.create_publisher(PoseArray, '/rebar_intersections', qos_reliable)
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_timer(0.033, self.viz_timer_cb)
        self.get_logger().info('[DETECTOR] Ready — pure-CV overlay mode.')

    # ─────────────────────────────────────────── callbacks
    def target_3d_cb(self, msg):
        pass   # kept for topic compatibility

    def index_cb(self, msg):
        with self.data_lock:
            self.active_index = msg.data

    def info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)

    def rgb_cb(self, msg):
        with self.data_lock:
            self.latest_image = msg

    # ─────────────────────────────────────────── visualisation (pure CV every frame)
    def viz_timer_cb(self):
        with self.data_lock:
            img_msg      = self.latest_image
            active_idx   = self.active_index
            pub_px       = list(self.published_pixels)   # scan-time pixels, index-aligned

        if img_msg is None:
            return

        try:
            cv_img  = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            vis_img = cv2.convertScaleAbs(cv_img, alpha=1.2, beta=30)
            h, w    = vis_img.shape[:2]

            # ── Detect intersections live on the current frame
            live_pts = self._detect_intersections_in(cv_img)

            # ── For the active index, find the closest live point to the
            #    scan-time pixel so the green box snaps to the right intersection
            active_scan_px = pub_px[active_idx] if 0 <= active_idx < len(pub_px) else None

            drawn_green = False
            for (u, v) in live_pts:
                # Is this the active intersection?
                is_active = False
                if active_scan_px is not None:
                    dist = np.linalg.norm(np.array([u, v]) - np.array(active_scan_px))
                    # 120-px radius: generous enough for the arm having moved a bit
                    if dist < 120:
                        is_active = True

                if is_active and not drawn_green:
                    color    = (0, 255, 0)
                    box_size = 30
                    thick    = 3
                    cv2.putText(vis_img, 'VISITING',
                                (u + 35, v - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    drawn_green = True
                else:
                    color    = (0, 0, 255)
                    box_size = 22
                    thick    = 2

                cv2.rectangle(vis_img,
                              (u - box_size, v - box_size),
                              (u + box_size, v + box_size),
                              color, thick)
                cv2.circle(vis_img, (u, v), 4, color, -1)

            # HUD
            cv2.putText(vis_img, 'REAL-TIME AR TRACKING',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_img, f'POINTS: {len(live_pts)}',
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            status = f'VISITING idx {active_idx}' if active_idx >= 0 else 'IDLE'
            cv2.putText(vis_img, status,
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if active_idx >= 0 else (180, 180, 180), 2)
            cv2.putText(vis_img, 'GREEN=active  RED=queued',
                        (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

            cv2.imshow('Robot Camera View', vis_img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'GUI Error: {e}')

    # ─────────────────────────────────────────── pure-CV intersection detector
    def _detect_intersections_in(self, bgr_img):
        """
        Detect rebar grid intersections directly in the given BGR frame.
        Returns list of (u, v) pixel centroids — no TF, no 3D.
        """
        gray    = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 50, 150)
        lines   = cv2.HoughLinesP(edges, 1, np.pi / 180, 50,
                                  minLineLength=80, maxLineGap=30)
        return self._intersections_from_lines(lines)

    def _intersections_from_lines(self, lines):
        if lines is None:
            return []
        h_lines, v_lines = [], []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 15 or abs(angle) > 165:
                h_lines.append(l[0])
            elif abs(abs(angle) - 90) < 15:
                v_lines.append(l[0])
        raw = []
        for hl in h_lines:
            for vl in v_lines:
                pt = self._line_intersect(hl, vl)
                if pt:
                    raw.append(pt)
        return self._cluster_points(raw, radius=50)

    # ─────────────────────────────────────────── detection trigger (for supervisor)
    def trigger_cb(self, msg):
        if msg.data:
            threading.Thread(target=self.detect_and_publish, daemon=True).start()

    def detect_and_publish(self):
        """
        Run once at scan time.  Converts detected pixel intersections to 3D
        base_link poses and publishes them for the supervisor to visit.
        Also saves the pixel positions index-aligned with the published poses
        so the visualiser can snap the green box to the right intersection.
        """
        with self.data_lock:
            if self.latest_image is None or self.K is None:
                return
            curr_msg = self.latest_image

        try:
            cv_img    = self.bridge.imgmsg_to_cv2(curr_msg, 'bgr8')
            pixel_pts = self._detect_intersections_in(cv_img)
            self.get_logger().info(f'[DETECTOR] {len(pixel_pts)} intersections found.')

            temp_3d, poses, saved_px = [], [], []
            for u, v in pixel_pts:
                pt_3d = self._pixel_to_3d(u, v)
                if pt_3d is not None:
                    temp_3d.append(pt_3d)
                    saved_px.append((u, v))
                    p = Pose()
                    p.position.x, p.position.y, p.position.z = map(float, pt_3d)
                    p.orientation.w = 1.0
                    poses.append(p)

            with self.data_lock:
                self.all_rebar_3d_points = temp_3d
                self.published_pixels    = saved_px   # index-aligned with poses

            pa = PoseArray()
            pa.header.frame_id = 'base_link'
            pa.header.stamp    = self.get_clock().now().to_msg()
            pa.poses           = poses
            self.pose_pub.publish(pa)
            self.get_logger().info(f'[DETECTOR] Published {len(poses)} poses.')

        except Exception as e:
            self.get_logger().error(f'Logic error: {e}')

    # ─────────────────────────────────────────── 3D back-projection (supervisor only)
    def _pixel_to_3d(self, u, v):
        try:
            tf      = self.tf_buffer.lookup_transform(
                          'base_link', 'ZED_X', rclpy.time.Time())
            ray_cam = np.array([
                (u - self.K[0, 2]) / self.K[0, 0],
                (v - self.K[1, 2]) / self.K[1, 1],
                -1.0,
            ])
            ray_cam /= np.linalg.norm(ray_cam)
            rot    = R.from_quat([
                tf.transform.rotation.x, tf.transform.rotation.y,
                tf.transform.rotation.z, tf.transform.rotation.w,
            ]).as_matrix()
            ray_base = rot @ ray_cam
            origin   = np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ])
            if abs(ray_base[1]) < 1e-6:
                return None
            s = (self.rebar_plane_y - origin[1]) / ray_base[1]
            return origin + s * ray_base if s > 0 else None
        except:
            return None

    # ─────────────────────────────────────────── geometry helpers
    @staticmethod
    def _line_intersect(l1, l2):
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1)))

    @staticmethod
    def _cluster_points(points, radius):
        if not points:
            return []
        used, clustered = [False] * len(points), []
        for i, p in enumerate(points):
            if used[i]:
                continue
            c = [p]
            used[i] = True
            for j in range(i + 1, len(points)):
                if not used[j] and \
                        np.linalg.norm(np.array(p) - np.array(points[j])) < radius:
                    c.append(points[j])
                    used[j] = True
            clustered.append((
                int(np.mean([x[0] for x in c])),
                int(np.mean([x[1] for x in c])),
            ))
        return clustered


def main():
    rclpy.init()
    node = RebarDetector()
    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()