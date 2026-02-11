# NuRec gRPC 调用流程

本文档说明如何不经过前端页面，直接通过 gRPC 调用 NuRec service，完成以下流程：

1. 连接 NuRec gRPC 服务
2. 查询可用 scene / camera / trajectory
3. 构造 `RGBRenderRequest`
4. 请求渲染并保存 JPEG 图片

适用于当前仓库中的 NuRec Web Viewer backend 调试和问题定位。

---

## 1. 前置条件

- NuRec 容器已启动并监听端口（默认 `46435`）
- 本机可访问 `localhost:46435`
- 已有可用 USDZ（并已由容器加载）
- Python 环境包含：
  - `grpcio`
  - `grpcio-tools`
  - `numpy`
  - `scipy`

---

## 2. 使用本仓库 proto 生成 Python stub

本仓库 proto 在 `proto/` 下：

- `proto/common.proto`
- `proto/sensorsim.proto`

可用以下命令生成（示例输出到 `backend/.cache/proto_generated`）：

```bash
python -m grpc_tools.protoc \
  -I./proto \
  --python_out=./backend/.cache/proto_generated \
  --grpc_python_out=./backend/.cache/proto_generated \
  ./proto/common.proto \
  ./proto/sensorsim.proto
```

---

## 3. gRPC 调用顺序（推荐）

建议按下面顺序调用，便于定位问题：

1. `get_available_scenes(Empty)`
2. `get_available_cameras(AvailableCamerasRequest(scene_id=...))`
3. `get_available_trajectories(AvailableTrajectoriesRequest(scene_id=...))`
4. `render_rgb(RGBRenderRequest(...))`

说明：

- `scene_id`：来自第 1 步返回
- `camera_intrinsics`：通常直接使用 `available_camera.intrinsics`
- `timestamp_us`：建议使用 trajectory 的时间范围内值
- `frame_end_us` 必须大于 `frame_start_us`（通常 `+1`）

---

## 4. 最小可运行示例

将下面代码保存为 `scripts/diag_nurec_grpc_render.py`（或临时直接运行）。

```python
import sys
import grpc

from scipy.spatial.transform import Rotation as R
import numpy as np

# 1) 修改为你的 stub 目录
PROTO_GEN_DIR = "./backend/.cache/proto_generated"
if PROTO_GEN_DIR not in sys.path:
    sys.path.insert(0, PROTO_GEN_DIR)

import common_pb2
import sensorsim_pb2
import sensorsim_pb2_grpc


def pose_to_matrix(pose):
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = R.from_quat([pose.quat.x, pose.quat.y, pose.quat.z, pose.quat.w]).as_matrix()
    m[:3, 3] = [pose.vec.x, pose.vec.y, pose.vec.z]
    return m


def matrix_to_pose(m):
    q = R.from_matrix(m[:3, :3]).as_quat()  # x y z w
    return common_pb2.Pose(
        vec=common_pb2.Vec3(x=float(m[0, 3]), y=float(m[1, 3]), z=float(m[2, 3])),
        quat=common_pb2.Quat(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
    )


def main():
    channel = grpc.insecure_channel(
        "127.0.0.1:46435",
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )
    stub = sensorsim_pb2_grpc.SensorsimServiceStub(channel)

    # 1) scene
    scenes = stub.get_available_scenes(common_pb2.Empty(), timeout=30)
    if not scenes.scene_ids:
        raise RuntimeError("No available scenes")
    scene_id = scenes.scene_ids[0]
    print("scene_id:", scene_id)

    # 2) camera
    cameras = stub.get_available_cameras(
        sensorsim_pb2.AvailableCamerasRequest(scene_id=scene_id),
        timeout=120,
    )
    if not cameras.available_cameras:
        raise RuntimeError("No available cameras")
    cam = cameras.available_cameras[0]
    print("camera:", cam.logical_id)

    # 3) trajectory
    trajectories = stub.get_available_trajectories(
        sensorsim_pb2.AvailableTrajectoriesRequest(scene_id=scene_id),
        timeout=30,
    )
    if not trajectories.available_trajectories:
        raise RuntimeError("No available trajectories")
    poses = trajectories.available_trajectories[0].trajectory.poses
    if not poses:
        raise RuntimeError("Trajectory has no poses")

    # 选一个中间时间戳，避免边界时间点
    mid_pose = poses[len(poses) // 2]
    timestamp_us = int(mid_pose.timestamp_us)
    rig_pose = mid_pose.pose

    # 4) 构造相机位姿：rig_pose * rig_to_camera
    rig_m = pose_to_matrix(rig_pose)
    cam_m = pose_to_matrix(cam.rig_to_camera)
    sensor_pose = matrix_to_pose(rig_m @ cam_m)

    # 使用相机原始 intrinsics，缩放分辨率以降低负载
    camera_spec = sensorsim_pb2.CameraSpec()
    camera_spec.CopyFrom(cam.intrinsics)
    camera_spec.resolution_h = max(1, int(camera_spec.resolution_h * 0.25))
    camera_spec.resolution_w = max(1, int(camera_spec.resolution_w * 0.25))

    req = sensorsim_pb2.RGBRenderRequest(
        scene_id=scene_id,
        resolution_h=camera_spec.resolution_h,
        resolution_w=camera_spec.resolution_w,
        camera_intrinsics=camera_spec,
        frame_start_us=timestamp_us,
        frame_end_us=timestamp_us + 1,  # 必须不同
        sensor_pose=sensorsim_pb2.PosePair(start_pose=sensor_pose, end_pose=sensor_pose),
        dynamic_objects=[],
        image_format=sensorsim_pb2.ImageFormat.JPEG,
        image_quality=90,
    )

    resp = stub.render_rgb(req, timeout=180)
    with open("nurec_render.jpg", "wb") as f:
        f.write(resp.image_bytes)
    print("saved: nurec_render.jpg, bytes:", len(resp.image_bytes))


if __name__ == "__main__":
    main()
```

运行：

```bash
python scripts/diag_nurec_grpc_render.py
```

---

## 5. 常见失败与排查建议

### A. `StatusCode.UNAVAILABLE`

- 现象：连不上 `localhost:46435`
- 排查：
  - NuRec 容器是否还在运行
  - 端口映射是否正确（`-p 46435:46435`）
  - 容器刚重启时可等待 10~30 秒再重试

### B. `StatusCode.UNKNOWN: NRenderer.render failed`

- 现象：scene/camera 能查到，但 `render_rgb` 失败
- 结论：通常是 NuRec 容器内部渲染链路问题，不是 REST 包装层问题
- 排查：
  - 直接用本文件示例进行 gRPC 直连验证（绕过 backend）
  - 查看 NuRec 容器日志中是否出现 `cannot get cuda resource` / `JIT compiling failed`

### C. `CUDA out of memory`

- 现象：加载 scene 或 render 时 OOM
- 建议：
  - 降低 `resolution_scale`（例如 `0.1`/`0.25`）
  - 减少并发渲染 camera 数量（先单 camera）
  - 重启容器释放显存
  - 在容器中设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## 6. 与本项目 backend 接口对应关系

本项目 backend 的 REST 封装基本映射为：

- `POST /api/load` -> `get_available_scenes + get_available_cameras + get_available_trajectories`
- `GET /api/scenario` -> 返回 scene/camera/time_range 元信息
- `POST /api/render` / `GET /api/render/{camera_id}` -> `render_rgb`

因此当页面“有连接但无图片”时，建议先跑本文件第 4 节的直连脚本，快速区分：

- **backend 参数拼接问题**
- 或 **NuRec service 内部渲染问题**

