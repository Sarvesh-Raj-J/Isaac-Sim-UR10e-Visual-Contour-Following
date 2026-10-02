#!/usr/bin/env python3
"""
seam_weld_detector.py
=====================
Camera vision node for seam welding on a UR10e (eye-in-hand).

Topics in:
  /rgb             → sensor_msgs/Image  (colour feed from Isaac Sim)
  /depth           → sensor_msgs/Image  (32FC1 depth in metres)
  /camera_info     → sensor_msgs/CameraInfo
  /detect_seams    → std_msgs/Bool      (trigger: run detection once)
  /active_seam_index → std_msgs/Int32  (which seam is being traversed)
  /scan_phase      → std_msgs/String    ("OVERVIEW" | "SCAN" | "IDLE")

Topics out:
  /seam_edges      → geometry_msgs/PoseArray
                     consecutive pairs: poses[0],poses[1] = edge0 start/end, ...
"""

import threading

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Int32, String

CAMERA_TF_FRAME = "ZED_X"
BASE_FRAME      = "base_link"

# Edge detection tuning
CANNY_LOW      = 40
CANNY_HIGH     = 120
MIN_LINE_PX    = 70
HOUGH_THRESH   = 40
HOUGH_GAP      = 15
MERGE_ANGLE    = 10.0   # degrees
MERGE_DIST     = 18.0   # pixels
MIN_EDGE_3D    = 0.03   # metres
MAX_EDGE_3D    = 1.20   # metres
DEPTH_PATCH    = 4      # half-patch for depth sampling


class SeamWeldDetector(Node):

    def __init__(self):
        super().__init__("seam_weld_detector")

        self.bridge        = CvBridge()
        self.K             = None
        self.latest_rgb    = None
        self.latest_depth  = None
        self.active_idx    = -1
        self.scan_phase    = "IDLE"
        self._lock         = threading.Lock()

        # Scan-time pixel segments, index-aligned with /seam_edges poses
        self.published_segments: list = []

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Image,      "/rgb",               self._rgb_cb,     10)
        self.create_subscription(Image,      "/depth",             self._depth_cb,   10)
        self.create_subscription(CameraInfo, "/camera_info",       self._info_cb,    10)
        self.create_subscription(Bool,       "/detect_seams",      self._trigger_cb, qos)
        self.create_subscription(Int32,      "/active_seam_index", self._active_cb,  10)
        self.create_subscription(String,     "/scan_phase",        self._phase_cb,   10)

        self.edge_pub = self.create_publisher(PoseArray, "/seam_edges", qos)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        cv2.namedWindow("Seam Weld Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Seam Weld Camera", 900, 506)

        self.create_timer(0.033, self._viz_cb)
        self.get_logger().info("[DETECTOR] Ready — subscribed to /rgb.")

    # ── Subscriptions ──────────────────────────────────────────────────────────
    def _rgb_cb(self, msg):
        with self._lock:
            self.latest_rgb = msg

    def _depth_cb(self, msg):
        with self._lock:
            self.latest_depth = msg

    def _info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)

    def _active_cb(self, msg):
        with self._lock:
            self.active_idx = msg.data

    def _phase_cb(self, msg):
        with self._lock:
            self.scan_phase = msg.data

    def _trigger_cb(self, msg):
        if msg.data:
            threading.Thread(target=self._detect_and_publish, daemon=True).start()

    # ── Live visualisation (30 Hz) ─────────────────────────────────────────────
    def _viz_cb(self):
        with self._lock:
            rgb_msg    = self.latest_rgb
            active_idx = self.active_idx
            pub_segs   = list(self.published_segments)
            phase      = self.scan_phase

        if rgb_msg is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            vis   = cv2.convertScaleAbs(frame, alpha=1.15, beta=20)
            h, w  = vis.shape[:2]

            if phase == "OVERVIEW":
                # Wide-angle overview — draw ALL detected edges lightly, no labels
                live_segs = self._detect_edges_in(frame)
                for (x1, y1), (x2, y2) in live_segs:
                    cv2.line(vis, (x1, y1), (x2, y2), (255, 200, 0), 1)
                self._hud(vis, "OVERVIEW — WIDE SCAN", phase, len(live_segs),
                          active_idx, h)

            else:
                # SCAN / IDLE — full annotated overlay
                live_segs = self._detect_edges_in(frame)
                for i, seg in enumerate(live_segs):
                    (x1, y1), (x2, y2) = seg
                    is_active = False
                    if 0 <= active_idx < len(pub_segs):
                        ps  = pub_segs[active_idx]
                        cx1 = (ps[0][0] + ps[1][0]) / 2
                        cy1 = (ps[0][1] + ps[1][1]) / 2
                        cx2 = (x1 + x2) / 2
                        cy2 = (y1 + y2) / 2
                        if np.hypot(cx2 - cx1, cy2 - cy1) < 100:
                            is_active = True
                    if is_active:
                        color, thick = (0, 255, 0), 4
                        mid = ((x1 + x2) // 2, (y1 + y2) // 2)
                        cv2.putText(vis, "WELDING",
                                    (mid[0] + 10, mid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                    else:
                        color, thick = (0, 80, 255), 2

                    cv2.line(vis, (x1, y1), (x2, y2), color, thick)
                    cv2.circle(vis, (x1, y1), 5, color, -1)
                    cv2.circle(vis, (x2, y2), 5, color, -1)
                    cv2.putText(vis, str(i),
                                ((x1 + x2) // 2, (y1 + y2) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                self._hud(vis, "SEAM WELD DETECTOR", phase, len(live_segs),
                          active_idx, h)

            cv2.imshow("Seam Weld Camera", vis)
            cv2.waitKey(1)

        except Exception as exc:
            self.get_logger().error(f"Viz error: {exc}")

    def _hud(self, vis, title, phase, n_edges, active_idx, h):
        phase_colour = {
            "OVERVIEW": (0, 200, 255),
            "SCAN":     (0, 255, 100),
            "IDLE":     (180, 180, 180),
        }.get(phase, (180, 180, 180))

        cv2.putText(vis, title,
                    (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        cv2.putText(vis, f"PHASE: {phase}",
                    (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.60, phase_colour, 2)
        cv2.putText(vis, f"EDGES LIVE: {n_edges}",
                    (20, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        status = (f"WELDING edge {active_idx}" if active_idx >= 0
                  else "IDLE")
        cv2.putText(vis, status,
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (0, 255, 0) if active_idx >= 0 else (180, 180, 180), 2)
        cv2.putText(vis, "GREEN=active  ORANGE=queued",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (180, 180, 180), 1)

    # ── One-shot detection triggered by supervisor ─────────────────────────────
    def _detect_and_publish(self):
        with self._lock:
            rgb_msg   = self.latest_rgb
            depth_msg = self.latest_depth

        if rgb_msg is None:
            self.get_logger().warn("[DETECTOR] No RGB image yet.")
            return
        if self.K is None:
            self.get_logger().warn("[DETECTOR] No camera_info yet.")
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = (self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
                     if depth_msg is not None else None)
        except Exception as exc:
            self.get_logger().error(f"cv_bridge: {exc}")
            return

        pixel_segs = self._detect_edges_in(bgr)
        self.get_logger().info(
            f"[DETECTOR] {len(pixel_segs)} image edges found."
        )

        poses: list[Pose] = []
        valid_segs: list  = []

        for i, ((u1, v1), (u2, v2)) in enumerate(pixel_segs):
            p3d1 = self._pixel_to_3d(u1, v1, depth)
            p3d2 = self._pixel_to_3d(u2, v2, depth)
            if p3d1 is None or p3d2 is None:
                self.get_logger().warn(
                    f"  seg {i}: 3D projection failed "
                    f"(depth None at ({u1},{v1}) or ({u2},{v2}))"
                )
                continue
            edge_len = float(np.linalg.norm(p3d2 - p3d1))
            self.get_logger().info(
                f"  seg {i}: p0={np.round(p3d1,3)} p1={np.round(p3d2,3)} "
                f"len={edge_len:.3f}m"
            )
            if not (MIN_EDGE_3D <= edge_len <= MAX_EDGE_3D):
                self.get_logger().warn(
                    f"  seg {i}: length {edge_len:.3f}m outside "
                    f"[{MIN_EDGE_3D},{MAX_EDGE_3D}] — filtered"
                )
                continue

            for pt in (p3d1, p3d2):
                p = Pose()
                p.position.x = float(pt[0])
                p.position.y = float(pt[1])
                p.position.z = float(pt[2])
                p.orientation.w = 1.0
                poses.append(p)
            valid_segs.append(((u1, v1), (u2, v2)))

        self.get_logger().info(
            f"[DETECTOR] {len(valid_segs)} valid 3D seam edges — publishing."
        )

        with self._lock:
            self.published_segments = valid_segs

        pa = PoseArray()
        pa.header.frame_id = BASE_FRAME
        pa.header.stamp    = self.get_clock().now().to_msg()
        pa.poses           = poses
        self.edge_pub.publish(pa)

    # ── CV edge detection ──────────────────────────────────────────────────────
    def _detect_edges_in(self, bgr):
        """Returns list of ((x1,y1),(x2,y2)) merged line segments."""
        gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        canny   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
        lines   = cv2.HoughLinesP(canny, 1, np.pi / 180,
                                  threshold=HOUGH_THRESH,
                                  minLineLength=MIN_LINE_PX,
                                  maxLineGap=HOUGH_GAP)
        if lines is None:
            return []
        raw = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if np.hypot(x2 - x1, y2 - y1) >= MIN_LINE_PX:
                raw.append(((x1, y1), (x2, y2)))
        return self._merge_lines(raw)

    def _merge_lines(self, segments):
        if not segments:
            return []
        used, merged = [False] * len(segments), []
        for i, seg in enumerate(segments):
            if used[i]:
                continue
            group, used[i] = [seg], True
            for j in range(i + 1, len(segments)):
                if not used[j] and self._segs_similar(seg, segments[j]):
                    group.append(segments[j])
                    used[j] = True
            pts = np.array([p for s in group for p in s])
            if len(pts) < 2:
                continue
            d2 = np.sum((pts[:, None] - pts[None, :]) ** 2, axis=2)
            i1, i2 = np.unravel_index(d2.argmax(), d2.shape)
            merged.append((tuple(pts[i1].tolist()), tuple(pts[i2].tolist())))
        return merged

    def _segs_similar(self, s1, s2):
        def _angle(seg):
            return np.degrees(np.arctan2(
                seg[1][1] - seg[0][1], seg[1][0] - seg[0][0]
            )) % 180.0

        if min(abs(_angle(s1) - _angle(s2)),
               180.0 - abs(_angle(s1) - _angle(s2))) > MERGE_ANGLE:
            return False
        dx = s2[1][0] - s2[0][0]
        dy = s2[1][1] - s2[0][1]
        L  = np.hypot(dx, dy)
        if L < 1e-6:
            return False
        mx = (s1[0][0] + s1[1][0]) / 2.0
        my = (s1[0][1] + s1[1][1]) / 2.0
        return abs((my - s2[0][1]) * dx - (mx - s2[0][0]) * dy) / L < MERGE_DIST

    # ── 3D back-projection ─────────────────────────────────────────────────────
    def _sample_depth(self, u, v, depth_img):
        h, w = depth_img.shape[:2]
        v0, v1 = max(0, v - DEPTH_PATCH), min(h, v + DEPTH_PATCH + 1)
        u0, u1 = max(0, u - DEPTH_PATCH), min(w, u + DEPTH_PATCH + 1)
        patch = depth_img[v0:v1, u0:u1]
        valid = patch[(patch > 0.05) & (patch < 8.0)
                      & ~np.isnan(patch) & ~np.isinf(patch)]
        return float(np.median(valid)) if len(valid) > 0 else None

    def _pixel_to_3d(self, u, v, depth_img) -> np.ndarray | None:
        if self.K is None:
            return None

        # Use depth if available
        d = None
        if depth_img is not None:
            d = self._sample_depth(u, v, depth_img)

        if d is None:
            return None

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        # ZED_X Z-axis points UP in world (+Z_world).
        # The camera looks in the -Z_ZED_X direction, so depth projects onto -Z.
        # (Confirmed by rebar_detector.py which uses z=-1.0 for the same frame.)
        pt_cam = np.array([
            (u - cx) * d / fx,
            (v - cy) * d / fy,
            -d,
        ])

        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, CAMERA_TF_FRAME, rclpy.time.Time()
            )
            rot = R.from_quat([
                tf.transform.rotation.x, tf.transform.rotation.y,
                tf.transform.rotation.z, tf.transform.rotation.w,
            ]).as_matrix()
            t = np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ])
            return rot @ pt_cam + t
        except Exception as exc:
            self.get_logger().warn(
                f"TF lookup failed: {exc}", throttle_duration_sec=5.0
            )
            return None


def main():
    rclpy.init()
    node = SeamWeldDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
