import jaxtyping as jt
import torch
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class CropData:
    x_start: int
    y_start: int
    x_end: int
    y_end: int


CropDataContainer = List[CropData]


@dataclass
class VideoPreProcessorOutput:
    video: jt.UInt8[torch.Tensor, "t c h_out w_out"]
    crop_data: CropDataContainer
    landmarks: jt.Float[np.ndarray, "t 478 2"]
    shape: jt.Int32[np.ndarray, "t 2"]

class VideoPreProcessor:
    def __init__(self, disable_video_preprocessor=False, crop_scale_factor=2, crop_type="per_frame", resize_size=512):
        self.disable_video_preprocessor = disable_video_preprocessor
        self.crop_scale_factor = crop_scale_factor
        self.crop_type = crop_type
        self.resize_size = resize_size

    def __call__(
        self,
        video: jt.UInt8[torch.Tensor, "t c h w"],
        landmarks: jt.Float[np.ndarray, "t 478 2"],
    ) -> VideoPreProcessorOutput:
        if self.disable_video_preprocessor:
            return self._preprocess_identity(video, landmarks)
        return self._preprocess_using_landmarks(video, landmarks)

    def _preprocess_identity(
        self,
        video: jt.UInt8[torch.Tensor, "t c h w"],
        landmarks: jt.Float[np.ndarray, "t 478 2"],
    ) -> VideoPreProcessorOutput:
        shape = np.array([video.shape[2], video.shape[3]])
        crop_data_container = [
            CropData(
                x_start=0,
                y_start=0,
                x_end=video.shape[2],
                y_end=video.shape[3],
            )
        ]
        return VideoPreProcessorOutput(
            video=video,
            landmarks=landmarks,
            shape=shape,
            crop_data=crop_data_container,
        )

    def _preprocess_using_landmarks(
        self,
        video: jt.UInt8[torch.Tensor, "t c h w"],
        landmarks: jt.Float[np.ndarray, "t 478 2"],
    ) -> VideoPreProcessorOutput:
        new_frames = []
        landmarks_container = []
        shape_container = []
        height, width = video.shape[2:]
        crop_data_container = self.get_crop_data(landmarks, height, width)
        for frame, lmk, crop_data in zip(
            video, landmarks, crop_data_container
        ):
            (
                cropped_frame,
                new_lmk,
                original_height,
                original_width,
            ) = self._crop_face(frame, lmk, crop_data)
            new_frames.append(cropped_frame)
            landmarks_container.append(new_lmk)
            shape_container.append(np.array([original_height, original_width]))
        return VideoPreProcessorOutput(
            video=torch.stack(new_frames),
            crop_data=crop_data_container,
            landmarks=np.stack(landmarks_container),
            shape=np.stack(shape_container),
        )

    def get_crop_data(
        self,
        landmarks: jt.Float[np.ndarray, "t 68 2"],
        height: int,
        width: int,
    ) -> CropDataContainer:
        crop_data_container = self._extract_crop_data(landmarks, height, width)
        crop_data_container = self._smooth_crop_data(crop_data_container)
        crop_data_container = self._refine_crop_data(
            crop_data_container, height, width
        )
        return crop_data_container

    def _extract_crop_data(
        self,
        landmarks: jt.Float[np.ndarray, "t 68 2"],
        height: int,
        width: int,
    ) -> CropDataContainer:
        crop_data_container = []
        for lmk in landmarks:
            min_x, min_y = np.min(lmk, axis=0)
            max_x, max_y = np.max(lmk, axis=0)

            crop_height = max_y - min_y
            crop_width = max_x - min_x
            crop_height = int(crop_height * self.crop_scale_factor)
            crop_width = int(crop_width * self.crop_scale_factor)

            crop_size = max(crop_height, crop_width)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            min_x = int(center_x - crop_size / 2)
            min_y = int(center_y - crop_size / 2)
            max_x = int(min_x + crop_size)
            max_y = int(min_y + crop_size)

            min_x, min_y = max(min_x, 0), max(min_y, 0)
            max_x, max_y = (
                min(max_x, width),
                min(max_y, height),
            )
            crop_data = CropData(
                x_start=min_x,
                y_start=min_y,
                x_end=max_x,
                y_end=max_y,
            )
            crop_data_container.append(crop_data)

        return crop_data_container

    def crop_single_frame(
        self,
        frame: jt.Float[torch.Tensor, "c h w"],
        lmk: jt.Float[np.ndarray, "68 2"],
        crop_data: CropData,
    ) -> Tuple[
        jt.Float[torch.Tensor, "c h_out w_out"],
        jt.Float[np.ndarray, "68 2"],
        int,
        int,
    ]:
        return self._crop_face(frame, lmk, crop_data)

    def _get_crop(
        self,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
        crop_size: int | None = None,
    ) -> Tuple[int, int, int, int]:
        if crop_size is None:
            cur_width = max_x - min_x
            cur_height = max_y - min_y
            crop_size = max(cur_width, cur_height)

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        min_x = int(center_x - crop_size / 2)
        min_y = int(center_y - crop_size / 2)
        max_x = int(min_x + crop_size)
        max_y = int(min_y + crop_size)

        return min_x, min_y, max_x, max_y

    def _recalculate_crop(
        self,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        if min_x < 0:
            max_x = max_x - min_x
            min_x = 0

        if min_y < 0:
            max_y = max_y - min_y
            min_y = 0

        if max_x > width:
            max_x = width
            crop_size = max_x - min_x
            min_x, min_y, max_x, max_y = self._get_crop(
                min_x, min_y, max_x, max_y, crop_size
            )

        if max_y > height:
            max_y = height
            crop_size = max_y - min_y
            min_x, min_y, max_x, max_y = self._get_crop(
                min_x, min_y, max_x, max_y, crop_size
            )

        return min_x, min_y, max_x, max_y

    def _refine_crop_data(
        self, crop_data: CropDataContainer, height: int, width: int
    ) -> CropDataContainer:
        new_data = []
        if self.crop_type == "union":
            min_x = np.min([data.x_start for data in crop_data])
            min_y = np.min([data.y_start for data in crop_data])
            max_x = np.max([data.x_end for data in crop_data])
            max_y = np.max([data.y_end for data in crop_data])

            min_x, min_y, max_x, max_y = self._get_crop(
                min_x, min_y, max_x, max_y
            )

            min_x, min_y, max_x, max_y = self._recalculate_crop(
                min_x, min_y, max_x, max_y, width, height
            )

            for _ in crop_data:
                new_data.append(
                    CropData(
                        x_start=min_x, y_start=min_y, x_end=max_x, y_end=max_y
                    )
                )
        elif self.crop_type == "per_frame":
            for data in crop_data:
                min_x, min_y, max_x, max_y = (
                    data.x_start,
                    data.y_start,
                    data.x_end,
                    data.y_end,
                )
                min_x, min_y, max_x, max_y = self._get_crop(
                    min_x, min_y, max_x, max_y
                )

                min_x, min_y, max_x, max_y = self._recalculate_crop(
                    min_x, min_y, max_x, max_y, width, height
                )
                new_data.append(
                    CropData(
                        x_start=min_x, y_start=min_y, x_end=max_x, y_end=max_y
                    )
                )
        else:
            raise ValueError(f"Unknown crop type: {self.crop_type}")
        return new_data

    def _smooth_crop_data(
        self, crop_data: CropDataContainer
    ) -> CropDataContainer:
        if self.crop_type == "union":
            return crop_data

        motion_threshold = 100
        smoothed_bboxes = []

        crop_data_array = np.array(
            [
                [data.x_start, data.y_start, data.x_end, data.y_end]
                for data in crop_data
            ]
        )
        for i in range(crop_data_array.shape[0]):
            current_bbox = crop_data_array[i]
            if i == 0:
                previous_smooth_bbox = current_bbox
                smoothed_bboxes.append(current_bbox)
                continue
            current_bbox = current_bbox
            previous_bbox = previous_smooth_bbox
            motion = np.linalg.norm(current_bbox - previous_bbox).item()
            # Adapt smoothing - less smoothing during fast movement
            alpha = 0.2 + 0.6 * min(1.0, motion / motion_threshold)
            smooth_bbox = (
                alpha * current_bbox + (1 - alpha) * previous_smooth_bbox
            )
            previous_smooth_bbox = smooth_bbox
            smoothed_bboxes.append(smooth_bbox)
        return [
            CropData(
                x_start=bbox[0], y_start=bbox[1], x_end=bbox[2], y_end=bbox[3]
            )
            for bbox in smoothed_bboxes
        ]

    def _crop_face(
        self,
        frame: jt.Float[torch.Tensor, "c h w"],
        lmk: jt.Float[np.ndarray, "68 2"],
        crop_data: CropData,
    ) -> Tuple[
        jt.Float[torch.Tensor, "c h_out w_out"],
        jt.Float[np.ndarray, "68 2"],
        int,
        int,
    ]:
        min_x, min_y, max_x, max_y = (
            crop_data.x_start,
            crop_data.y_start,
            crop_data.x_end,
            crop_data.y_end,
        )
        cropped_frame = frame[:, min_y:max_y, min_x:max_x]
        original_height, original_width = cropped_frame.shape[1:]
        cropped_frame = torch.nn.functional.interpolate(
            cropped_frame[None],
            (self.resize_size, self.resize_size),
            mode="bilinear",
            align_corners=False,
        )[0]
        width_ratio = self.resize_size / original_width
        height_ratio = self.resize_size / original_height
        new_lmk = lmk - np.array([min_x, min_y])
        new_lmk[:, 0] = new_lmk[:, 0] * width_ratio
        new_lmk[:, 1] = new_lmk[:, 1] * height_ratio

        new_lmk[:, 0] = np.clip(new_lmk[:, 0], 0, self.resize_size - 1)
        new_lmk[:, 1] = np.clip(new_lmk[:, 1], 0, self.resize_size - 1)
        return cropped_frame, new_lmk, original_height, original_width