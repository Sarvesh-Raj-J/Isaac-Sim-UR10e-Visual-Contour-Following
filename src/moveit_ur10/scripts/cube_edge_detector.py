#!/usr/bin/env python3
"""
cube_edge_detector.py
=====================
Camera vision node for cube edge traversal on UR10e (eye-in-hand).

Detects the largest rectangular face of a cube in the image, back-projects
its 4 corners to 3D via depth + TF, estimates the 4 hidden back-face corners
using the face normal, then publishes all 12 cube edges as a PoseArray.

Topics in:
  /rgb              sensor_msgs/Image   — colour feed from IsaacSim
  /depth            sensor_msgs/Image   — 32FC1 depth in metres
  /camera_info      sensor_msgs/CameraInfo
  /detect_cube      std_msgs/Bool       — one-shot trigger from supervisor
  /active_cube_edge std_msgs/Int32      — which edge is being traversed (for overlay)

Topics out:
  /cube_edges  geometry_msgs/PoseArray
               consecutive pairs: poses[2i], poses[2i+1] = edge i start/end
               edge order:  0-3 = front face, 4-7 = pillars, 8-11 = back face
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
from std_msgs.msg import Bool, Int32

CAMERA_FRAME  = "ZED_X"
BASE_FRAME    = "base_link"
DEPTH_PATCH   = 5      # half-side of depth-sampling patch (pixels)
MIN_FACE_AREA = 4000   # px² — ignore tiny rectangles


class CubeEdgeDetector(Node):

    def __init__(self):
        super().__init__("cube_edge_detector")

        self.bridge       = CvBridge()
        self.K            = None
        self.latest_rgb   = None
        self.latest_depth = None
        self.active_idx   = -1
        self._lock        = threading.Lock()

        # Scan-time pixel midpoints for each edge, index-aligned with /cube_edges
        self.published_midpoints: list = []

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Image,      "/rgb",              self._rgb_cb,     10)
        self.create_subscription(Image,      "/depth",            self._depth_cb,   10)
        self.create_subscription(CameraInfo, "/camera_info",      self._info_cb,    10)
        self.create_subscription(Bool,       "/detect_cube",      self._trigger_cb, qos)
        self.create_subscription(Int32,      "/active_cube_edge", self._active_cb,  10)

        self.edge_pub = self.create_publisher(PoseArray, "/cube_edges", qos)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        cv2.namedWindow("Cube Edge Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cube Edge Camera", 900, 506)

        self.create_timer(0.033, self._viz_cb)
        self.get_logger().info("[CUBE_DETECTOR] Ready — waiting for /detect_cube trigger.")

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

    def _trigger_cb(self, msg):
        if msg.data:
            threading.Thread(target=self._detect_and_publish, daemon=True).start()

    # ── Live visualisation (30 Hz) ─────────────────────────────────────────────
    def _viz_cb(self):
        with self._lock:
            rgb_msg    = self.latest_rgb
            active_idx = self.active_idx
            pub_mids   = list(self.published_midpoints)

        if rgb_msg is None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            vis   = cv2.convertScaleAbs(frame, alpha=1.15, beta=20)
            h, w  = vis.shape[:2]

            corners = self._detect_cube_face(frame)

            if corners is not None:
                for i in range(4):
                    p1 = tuple(corners[i].astype(int))
                    p2 = tuple(corners[(i + 1) % 4].astype(int))
                    cv2.line(vis, p1, p2, (0, 180, 255), 2)
                    cv2.circle(vis, p1, 6, (0, 255, 255), -1)
                    cv2.putText(vis, str(i), (p1[0] + 6, p1[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Highlight active edge midpoint from scan-time data
            if 0 <= active_idx < len(pub_mids):
                mid = pub_mids[active_idx]
                if mid is not None:
                    cv2.circle(vis, mid, 12, (0, 255, 0), -1)
                    cv2.putText(vis, "TRAVERSING",
                                (mid[0] + 15, mid[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # HUD
            cv2.putText(vis, "CUBE EDGE DETECTOR",
                        (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            face_txt = "FACE DETECTED" if corners is not None else "SEARCHING..."
            cv2.putText(vis, face_txt, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                        (0, 255, 100) if corners is not None else (0, 100, 255), 2)
            status = f"TRAVERSING edge {active_idx}" if active_idx >= 0 else "IDLE"
            cv2.putText(vis, status, (20, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if active_idx >= 0 else (180, 180, 180), 2)
            cv2.putText(vis, f"EDGES PUBLISHED: {len(pub_mids)}",
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
            cv2.putText(vis, "GREEN=active  ORANGE=face corners",
                        (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

            cv2.imshow("Cube Edge Camera", vis)
            cv2.waitKey(1)

        except Exception as exc:
            self.get_logger().error(f"Viz error: {exc}")

    # ── One-shot detection triggered by supervisor ─────────────────────────────
    def _detect_and_publish(self):
        with self._lock:
            rgb_msg   = self.latest_rgb
            depth_msg = self.latest_depth

        if rgb_msg is None or self.K is None:
            self.get_logger().warn("[CUBE_DETECTOR] No image or camera_info yet.")
            return
        try:
            bgr   = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = (self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
                     if depth_msg is not None else None)
        except Exception as exc:
            self.get_logger().error(f"cv_bridge: {exc}")
            return

        corners_px = self._detect_cube_face(bgr)
        if corners_px is None:
            self.get_logger().warn("[CUBE_DETECTOR] No cube face detected in image.")
            return

        self.get_logger().info("[CUBE_DETECTOR] Cube face found — projecting corners to 3D.")
        front_3d = self._corners_to_3d(corners_px, depth)
        if front_3d is None:
            self.get_logger().warn("[CUBE_DETECTOR] 3D projection failed.")
            return

        back_3d = self._estimate_back_face(front_3d)
        if back_3d is None:
            self.get_logger().warn("[CUBE_DETECTOR] Back face estimation failed.")
            return

        edges_3d = self._make_12_edges(front_3d, back_3d)
        self.get_logger().info(f"[CUBE_DETECTOR] {len(edges_3d)} cube edges generated.")

        poses: list[Pose] = []
        midpoints: list   = []
        for p0, p1 in edges_3d:
            for pt in (p0, p1):
                po = Pose()
                po.position.x = float(pt[0])
                po.position.y = float(pt[1])
                po.position.z = float(pt[2])
                po.orientation.w = 1.0
                poses.append(po)
            mid_px = self._project_to_pixel(np.mean([p0, p1], axis=0))
            midpoints.append(mid_px)
            self.get_logger().info(
                f"  edge: {np.round(p0, 3)} → {np.round(p1, 3)}"
            )

        with self._lock:
            self.published_midpoints = midpoints

        pa = PoseArray()
        pa.header.frame_id = BASE_FRAME
        pa.header.stamp    = self.get_clock().now().to_msg()
        pa.poses           = poses
        self.edge_pub.publish(pa)
        self.get_logger().info(
            f"[CUBE_DETECTOR] Published {len(edges_3d)} edges → /cube_edges."
        )

    # ── Cube face detection ────────────────────────────────────────────────────
    def _detect_cube_face(self, bgr) -> np.ndarray | None:
        """Return ordered (4,2) float32 corners of the largest rectangular face, or None."""
        gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        canny   = cv2.Canny(blurred, 40, 120)
        kernel  = np.ones((3, 3), np.uint8)
        canny   = cv2.dilate(canny, kernel, iterations=1)

        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best, best_area = None, MIN_FACE_AREA
        for cnt in contours:
            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > best_area:
                    best_area = area
                    best = approx.reshape(4, 2).astype(np.float32)

        return None if best is None else self._order_corners(best)

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """Order 4 corners: [top-left, top-right, bottom-right, bottom-left]."""
        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()
        tl   = pts[np.argmin(s)]
        br   = pts[np.argmax(s)]
        tr   = pts[np.argmin(diff)]
        bl   = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    # ── Depth sampling & 3D back-projection ───────────────────────────────────
    def _sample_depth(self, u: int, v: int, depth_img: np.ndarray) -> float | None:
        h, w   = depth_img.shape[:2]
        v0, v1 = max(0, v - DEPTH_PATCH), min(h, v + DEPTH_PATCH + 1)
        u0, u1 = max(0, u - DEPTH_PATCH), min(w, u + DEPTH_PATCH + 1)
        patch  = depth_img[v0:v1, u0:u1]
        valid  = patch[(patch > 0.05) & (patch < 8.0)
                       & ~np.isnan(patch) & ~np.isinf(patch)]
        return float(np.median(valid)) if len(valid) > 0 else None

    def _pixel_to_3d(self, u, v, d, tf_msg) -> np.ndarray:
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        # ZED_X looks along -Z_cam (same convention as seam_weld_detector)
        pt_cam = np.array([(u - cx) * d / fx,
                           (v - cy) * d / fy,
                           -d])
        rot = R.from_quat([
            tf_msg.transform.rotation.x, tf_msg.transform.rotation.y,
            tf_msg.transform.rotation.z, tf_msg.transform.rotation.w,
        ]).as_matrix()
        t = np.array([tf_msg.transform.translation.x,
                      tf_msg.transform.translation.y,
                      tf_msg.transform.translation.z])
        return rot @ pt_cam + t

    def _corners_to_3d(self, corners_px: np.ndarray, depth_img) -> list | None:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                BASE_FRAME, CAMERA_FRAME, rclpy.time.Time()
            )
        except Exception as exc:
            self.get_logger().warn(f"TF lookup failed: {exc}")
            return None

        result = []
        for (u, v) in corners_px:
            u, v = int(u), int(v)
            d = self._sample_depth(u, v, depth_img) if depth_img is not None else None
            if d is None:
                self.get_logger().warn(f"  No valid depth at corner ({u},{v})")
                return None
            result.append(self._pixel_to_3d(u, v, d, tf_msg))
            self.get_logger().info(f"  corner ({u},{v}) d={d:.3f}m → {np.round(result[-1], 3)}")
        return result

    # ── Back-face estimation ───────────────────────────────────────────────────
    def _estimate_back_face(self, front_3d: list) -> list | None:
        fc = np.array(front_3d)  # (4, 3)

        # Face normal from two non-parallel edges
        normal = np.cross(fc[1] - fc[0], fc[3] - fc[0])
        n_len  = np.linalg.norm(normal)
        if n_len < 1e-6:
            self.get_logger().warn("Degenerate face — zero normal.")
            return None
        normal /= n_len

        # Normal must point AWAY from camera (into the cube)
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                BASE_FRAME, CAMERA_FRAME, rclpy.time.Time()
            )
        except Exception as exc:
            self.get_logger().warn(f"TF lookup for normal: {exc}")
            return None
        cam_pos = np.array([tf_msg.transform.translation.x,
                            tf_msg.transform.translation.y,
                            tf_msg.transform.translation.z])
        if np.dot(normal, fc.mean(axis=0) - cam_pos) < 0:
            normal = -normal

        # Cube depth ≈ average side length (cube ⟹ all sides equal)
        side_lens  = [np.linalg.norm(fc[(i + 1) % 4] - fc[i]) for i in range(4)]
        cube_depth = float(np.mean(side_lens))
        self.get_logger().info(
            f"  Cube side ≈ {cube_depth:.3f} m  "
            f"normal={np.round(normal, 3)}"
        )

        return [pt + normal * cube_depth for pt in front_3d]

    # ── Generate 12 edges from 8 vertices ────────────────────────────────────
    @staticmethod
    def _make_12_edges(front: list, back: list) -> list:
        """
        front / back: lists of 4 np.ndarray in order [TL, TR, BR, BL].
        Returns 12 (start, end) pairs:
          0-3  front face ring
          4-7  vertical pillars (front→back)
          8-11 back face ring
        """
        fc, bc = front, back
        edges = []
        for i in range(4):                       # front face
            edges.append((fc[i], fc[(i + 1) % 4]))
        for i in range(4):                       # pillars
            edges.append((fc[i], bc[i]))
        for i in range(4):                       # back face
            edges.append((bc[i], bc[(i + 1) % 4]))
        return edges

    # ── Project 3D → pixel (for overlay midpoints) ────────────────────────────
    def _project_to_pixel(self, pt_3d: np.ndarray):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                CAMERA_FRAME, BASE_FRAME, rclpy.time.Time()
            )
        except Exception:
            return (0, 0)
        rot = R.from_quat([
            tf_msg.transform.rotation.x, tf_msg.transform.rotation.y,
            tf_msg.transform.rotation.z, tf_msg.transform.rotation.w,
        ]).as_matrix()
        t = np.array([tf_msg.transform.translation.x,
                      tf_msg.transform.translation.y,
                      tf_msg.transform.translation.z])
        pt_cam = rot @ pt_3d + t
        if pt_cam[2] >= 0:          # behind camera
            return (0, 0)
        d  = -pt_cam[2]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        return (int(pt_cam[0] * fx / d + cx),
                int(pt_cam[1] * fy / d + cy))


def main():
    rclpy.init()
    node = CubeEdgeDetector()
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
