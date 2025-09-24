# unigaze/infer_runtime.py
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import face_alignment
import numpy as np
import torch

# --- UniGaze / your repo imports (these exist in your repo) ---
from datasets.helper.image_transform import wrap_transforms
from gazelib.gaze.gaze_utils import pitchyaw_to_vector, vector_to_pitchyaw
from gazelib.gaze.normalize import estimateHeadPose, normalize
from gazelib.label_transform import get_face_center_by_nose
from omegaconf import OmegaConf
from utils import instantiate_from_cfg
from utils.util import set_seed

# ---------------- Helpers copied from predict_gaze_video.py ----------------


def draw_gaze(image_in, pitchyaw, thickness=8, color=(0, 0, 255)):
    image_out = image_in.copy()
    (h, w) = image_in.shape[:2]
    length = w / 2.0
    pos = (int(h / 2.0), int(w / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    end_point = (int(pos[0] + dx), int(pos[1] + dy))

    shadow_offset = 2
    shadow_color = (40, 40, 40)
    shadow_end = (end_point[0] + shadow_offset, end_point[1] + shadow_offset)
    cv2.arrowedLine(
        image_out,
        (pos[0] + shadow_offset, pos[1] + shadow_offset),
        shadow_end,
        shadow_color,
        thickness + 2,
        cv2.LINE_AA,
        tipLength=0.3,
    )

    thickness_values = [4, 3, 2, 1]
    num_layers = len(thickness_values)
    for i in range(num_layers):
        alpha = i / num_layers
        layer_color = tuple(int((1 - alpha) * color[j] + alpha * 255) for j in range(3))
        cv2.arrowedLine(
            image_out,
            pos,
            end_point,
            layer_color,
            thickness_values[i],
            cv2.LINE_AA,
            tipLength=0.3,
        )
    return image_out


def set_dummy_camera_model(image=None):
    h, w = image.shape[:2]
    focal_length = w * 4
    center = (w // 2, h // 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )
    camera_distortion = np.zeros((1, 5))
    return np.array(camera_matrix), np.array(camera_distortion)


def denormalize_predicted_gaze(gaze_yaw_pitch, R_inv):
    pred_gaze_cancel_nor = pitchyaw_to_vector(gaze_yaw_pitch.reshape(1, 2)).reshape(
        3, 1
    )
    pred_gaze_cancel_nor = np.matmul(R_inv, pred_gaze_cancel_nor.reshape(3, 1))
    pred_gaze_cancel_nor = pred_gaze_cancel_nor / np.linalg.norm(pred_gaze_cancel_nor)
    pred_yaw_pitch_cancel_nor = vector_to_pitchyaw(pred_gaze_cancel_nor.reshape(1, 3))
    return pred_gaze_cancel_nor, pred_yaw_pitch_cancel_nor


def load_checkpoint(model, ckpt_key, ckpt_path):
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    weights = torch.load(ckpt_path, map_location="cpu")
    model_state = weights[ckpt_key]
    if next(iter(model_state.keys())).startswith("module."):
        model_state = {k[7:]: v for k, v in model_state.items()}
    model.load_state_dict(model_state, strict=True)
    del weights


# ---------------- Runtime (model + pre/post) ----------------


class UniGazeRuntime:
    def __init__(self, cfg_path: str, ckpt_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        torch.set_grad_enabled(False)
        set_seed(42)

        pretrained_model_cfg = OmegaConf.load(cfg_path)["net_config"]
        pretrained_model_cfg.params.custom_pretrained_path = None

        self.model = instantiate_from_cfg(pretrained_model_cfg)
        load_checkpoint(self.model, "model_state", ckpt_path)
        self.model.eval().to(self.device)

        self.transform = wrap_transforms("basic_imagenet", image_size=224)
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=self.device.type,  # 'cpu' or 'cuda'
        )

        # Constants from your script
        self.focal_norm = 960
        self.distance_norm = 600
        self.roi_size = (224, 224)
        self.resize_factor = 0.5
        self.arrow_colors = [(47, 255, 173)]  # BGR

    # ---- One-frame inference on a BGR frame; returns annotated BGR frame ----
    def process_frame(self, image_original_bgr: np.ndarray) -> np.ndarray:
        image_original = image_original_bgr.copy()

        # resize for detection
        if self.resize_factor >= 1:
            image_resize = image_original.copy()
        else:
            image_resize = cv2.resize(
                image_original,
                dsize=None,
                fx=self.resize_factor,
                fy=self.resize_factor,
                interpolation=cv2.INTER_AREA,
            )

        image_resize = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
        preds = self.fa.get_landmarks(image_resize)

        # no face: just return original
        if preds is None:
            return image_original

        # ----- keep same semantics as predict_gaze_video.py -----
        landmarks_record = {}  # only add when a valid vector is produced
        vector_start_end_point_list = {}  # start/end 2D points for arrows
        bbox_record = {}  # for drawing rectangles (same idx keys)

        for idx, landmarks_in_original in enumerate(preds):
            color = self.arrow_colors[idx % len(self.arrow_colors)]

            # scale landmarks back to original size
            landmarks_in_original = landmarks_in_original / self.resize_factor
            x_min = int(landmarks_in_original[:, 0].min())
            x_max = int(landmarks_in_original[:, 0].max())
            y_min = int(landmarks_in_original[:, 1].min())
            y_max = int(landmarks_in_original[:, 1].max())

            # bbox for drawing (scale 1.2)
            scale_factor_draw = 1.2
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            bbox_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
            x_min_draw = max(
                0, bbox_center[0] - int(bbox_width * scale_factor_draw // 2)
            )
            x_max_draw = min(
                image_original.shape[1],
                bbox_center[0] + int(bbox_width * scale_factor_draw // 2),
            )
            y_min_draw = max(
                0, bbox_center[1] - int(bbox_height * scale_factor_draw // 2)
            )
            y_max_draw = min(
                image_original.shape[0],
                bbox_center[1] + int(bbox_height * scale_factor_draw // 2),
            )
            bbox_record[idx] = (x_min_draw, y_min_draw, x_max_draw, y_max_draw)

            # crop for normalization & inference (scale 2.0)
            scale_factor_crop = 2.0
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            bbox_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
            x_min_c = max(0, bbox_center[0] - int(bbox_width * scale_factor_crop // 2))
            x_max_c = min(
                image_original.shape[1],
                bbox_center[0] + int(bbox_width * scale_factor_crop // 2),
            )
            y_min_c = max(0, bbox_center[1] - int(bbox_height * scale_factor_crop // 2))
            y_max_c = min(
                image_original.shape[0],
                bbox_center[1] + int(bbox_height * scale_factor_crop // 2),
            )

            image = image_original[y_min_c:y_max_c, x_min_c:x_max_c]
            landmarks = landmarks_in_original - np.array([x_min_c, y_min_c])

            # camera + head pose
            camera_matrix, camera_distortion = set_dummy_camera_model(image=image)
            face_model_load = np.loadtxt("data/face_model.txt")
            face_model = face_model_load[[20, 23, 26, 29, 15, 19], :]
            facePts = face_model.reshape(6, 1, 3)

            landmarks_sub = (
                landmarks[[36, 39, 42, 45, 31, 35], :].astype(float).reshape(6, 1, 2)
            )
            hr, ht = estimateHeadPose(
                landmarks_sub, facePts, camera_matrix, camera_distortion
            )
            hR = cv2.Rodrigues(hr)[0]
            face_center_camera_cord, _ = get_face_center_by_nose(
                hR=hR, ht=ht, face_model_load=face_model_load
            )

            # normalize
            img_normalized, R, hR_norm, _, _, _ = normalize(
                image,
                landmarks,
                self.focal_norm,
                self.distance_norm,
                self.roi_size,
                face_center_camera_cord,
                hr,
                ht,
                camera_matrix,
                gc=None,
            )

            # skip bad heads (same as script)
            hr_norm = np.array(
                [np.arcsin(hR_norm[1, 2]), np.arctan2(hR_norm[0, 2], hR_norm[2, 2])]
            )
            if np.linalg.norm(hr_norm) > 80 * np.pi / 180:
                continue

            # inference
            input_var = img_normalized[:, :, [2, 1, 0]]  # BGR->RGB
            input_var = self.transform(input_var)
            input_var = torch.as_tensor(
                input_var, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                ret = self.model(input_var)

            pred_gaze = ret["pred_gaze"][0]
            pred_gaze_np = pred_gaze.detach().cpu().numpy()

            # denormalize to original camera coords, then project to 2D
            R_inv = np.linalg.inv(R)
            pred_gaze_cancel_nor, _ = denormalize_predicted_gaze(pred_gaze_np, R_inv)

            vec_length = pred_gaze_cancel_nor * -112 * 1.5
            gazeRay = np.concatenate(
                (
                    face_center_camera_cord.reshape(1, 3),
                    (face_center_camera_cord + vec_length).reshape(1, 3),
                ),
                axis=0,
            )
            result = cv2.projectPoints(
                gazeRay,
                np.array([0, 0, 0]).reshape(3, 1).astype(float),
                np.array([0, 0, 0]).reshape(3, 1).astype(float),
                camera_matrix,
                camera_distortion,
            )[0].reshape(2, 2)
            result += np.array([x_min_c, y_min_c])

            vector_start_point = (int(result[0][0]), int(result[0][1]))
            vector_end_point = (int(result[1][0]), int(result[1][1]))

            # only record these after a valid vector exists
            vector_start_end_point_list[idx] = (vector_start_point, vector_end_point)
            landmarks_record[idx] = landmarks_in_original

        # If nothing valid was produced, return original frame
        if not landmarks_record:
            return image_original

        # ----- draw exactly like predict_gaze_video.py (iterate over landmarks_record) -----
        for idx in list(landmarks_record.keys()):
            x_min_d, y_min_d, x_max_d, y_max_d = bbox_record[idx]
            color = self.arrow_colors[idx % len(self.arrow_colors)]

            cv2.rectangle(
                image_original, (x_min_d, y_min_d), (x_max_d, y_max_d), (0, 0, 240), 2
            )

            vsp, vep = vector_start_end_point_list[idx]
            shadow_offset = 2
            shadow_color = (40, 40, 40)
            shadow_end = (vep[0] + shadow_offset, vep[1] + shadow_offset)
            cv2.arrowedLine(
                image_original,
                (vsp[0] + shadow_offset, vsp[1] + shadow_offset),
                shadow_end,
                shadow_color,
                5,
                cv2.LINE_AA,
                tipLength=0.2,
            )

            thickness_values = [x * 3 for x in [4, 3, 2, 1]]
            num_layers = len(thickness_values)
            for i in range(num_layers):
                alpha = i / num_layers
                layer_color = tuple(
                    int((1 - alpha) * color[j] + alpha * 255) for j in range(3)
                )
                cv2.arrowedLine(
                    image_original,
                    vsp,
                    vep,
                    layer_color,
                    thickness_values[i],
                    cv2.LINE_AA,
                    tipLength=0.2,
                )

        return image_original

    # ---- Public APIs ----

    def predict_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """Accepts an RGB image (HxWx3) and returns an annotated RGB image."""
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        out_bgr = self.process_frame(bgr)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return out_rgb

    def predict_video(
        self, video_path: str
    ) -> Tuple[Optional[str], Optional[np.ndarray], float]:
        """
        Process a video file and return:
          - temp MP4 path (string) for Gradio Video
          - last annotated RGB frame (numpy) for Gradio Image
          - total runtime seconds (float)
        """
        t0 = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, 0.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)) or 25)

        tmp_mp4 = Path(tempfile.mkdtemp(prefix="unigaze_vid_")) / "out.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_mp4), fourcc, fps, (width, height))

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            out_bgr = self.process_frame(frame_bgr)
            writer.write(out_bgr)

        cap.release()
        writer.release()
        return str(tmp_mp4), float(time.time() - t0)

    def get_feature(self, image: np.ndarray) -> np.ndarray:
        """
        处理单张图片并返回高维特征向量。
        此方法完整复刻了 predict_image 的预处理逻辑，实现了完全的独立性。
        """

        # 1. 在方法内部定义它自己需要的所有工具和变量，不依赖外部修改
        def _set_dummy_camera_model(image_shape):
            h, w = image_shape[:2]
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array(
                [
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1],
                ],
                dtype="double",
            )
            return camera_matrix, np.zeros((1, 5))

        # 加载 3D 面部模型
        face_model_load = np.loadtxt(
            os.path.join(os.path.dirname(__file__), "data/face_model.txt")
        )
        # 创建图像转换器
        image_torch_transform = wrap_transforms("basic_imagenet", image_size=224)

        # 2. 人脸检测
        landmarks = self.fa.get_landmarks(image)
        if landmarks is None:
            return None

        # 默认只处理检测到的第一个人脸
        landmark = landmarks[0]

        # 3. 完整的预处理流程 (100% 复刻自 predict_image)
        crop_scale = 1.6
        x_min, y_min = landmark.min(axis=0)
        x_max, y_max = landmark.max(axis=0)
        w, h = x_max - x_min, y_max - y_min
        center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        x_min_crop = int(max(0, center[0] - w * crop_scale / 2))
        x_max_crop = int(min(image.shape[1], center[0] + w * crop_scale / 2))
        y_min_crop = int(max(0, center[1] - h * crop_scale / 2))
        y_max_crop = int(min(image.shape[0], center[1] + h * crop_scale / 2))

        img_cropped = image[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
        landmarks_cropped = landmark - np.array([x_min_crop, y_min_crop])

        cam_matrix, cam_dist = _set_dummy_camera_model(img_cropped.shape)

        landmarks_sub = landmarks_cropped[[36, 39, 42, 45, 31, 35], :]
        face_model_sub = face_model_load[[36, 39, 42, 45, 31, 35], :]
        hr, ht = estimateHeadPose(
            landmarks_sub.reshape(6, 1, 2),
            face_model_sub.reshape(6, 1, 3),
            cam_matrix,
            cam_dist,
        )

        face_center, _ = get_face_center_by_nose(
            cv2.Rodrigues(hr)[0], ht, face_model_load
        )

        img_norm, _, _, _, _, _ = normalize(
            img_cropped,
            landmarks_cropped,
            960,
            600,
            (224, 224),
            face_center,
            hr,
            ht,
            cam_matrix,
        )

        # 4. 提取并返回特征
        with torch.no_grad():
            img_tensor = (
                image_torch_transform(img_norm[:, :, ::-1].copy())
                .unsqueeze(0)
                .to(self.device)
            )
            features = self.model.vit.forward_features(img_tensor)
            feature_vector = features.squeeze().cpu().numpy()
            return feature_vector
