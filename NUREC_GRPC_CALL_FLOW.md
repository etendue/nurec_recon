# NuRec Backend / gRPC / Playback 调用流程（2026-02-12 更新）

本文档已按当前代码更新，覆盖以下内容：

1. Backend 如何读取 USDZ（已接入 `scenario.py + track.py` 同逻辑）
2. Backend 与 NuRec gRPC 的真实调用边界
3. 前端 play/stop/tick 的推荐对接方式
4. 联调与排障路径

---

## 1. 当前架构结论

### 1.1 Backend 数据来源（关键变更）

Backend 现在分成两条链路：

- **场景/轨迹/相机元数据链路**：本地直接解析 USDZ（不依赖 NuRec 元数据接口）
  - 入口：`backend/scenario_runtime.py`
  - 核心对象：`Scenario`、`Track`、`Tracks`、`PlaybackClock`
- **渲染链路**：仍通过 NuRec gRPC 的 `render_rgb()`
  - 入口：`backend/main.py` 的 `/api/render`、`/api/render/{camera_id}`

也就是说，`/api/load` 会做两件事：

1. 建立 gRPC 连接（用于渲染）
2. 本地解析 USDZ 并构建 runtime scenario（用于时间轴、pose、trajectory、playback）

### 1.2 与旧实现的区别

旧文档中的：

- `get_available_scenes`
- `get_available_cameras`
- `get_available_trajectories`

不再是 backend 主流程依赖。当前 backend 的 camera/trajectory/time range 来自 USDZ 本地解析结果。

---

## 2. 后端核心模块与职责

### 2.1 `backend/scenario_runtime.py`

该文件是从 `sample_code/scenario.py` + `sample_code/track.py` 抽取并适配后的运行时模块（去掉 `carla` 依赖）：

- `PoseType` / `InterpolatedPoses` / `Track`
  - 支持 `XYZ_QUAT` 与 `4x4 matrix` 互转
  - 平移线性插值 + 旋转 slerp 插值
- `Tracks`
  - 维护 `current_time`、`active_tracks`
  - `update(delta_us)` 返回 `new_tracks` / `removed_tracks`
- `Scenario`
  - 加载 `rig_trajectories.json`、`sequence_tracks.json`、`data_info.json`
  - 构建 `ego_poses`、`spectator`、`camera_calibrations`、`controllable_tracks`
- `PlaybackClock`
  - 后端播放状态机：`play()` / `stop()` / `reset()` / `tick()`

### 2.2 `backend/main.py`

- 负责 FastAPI 路由与 NuRec gRPC 渲染调用
- `/api/load` 之后会持有：
  - `app_state.runtime_scenario`
  - `app_state.playback`
- `pose/trajectory/render` 使用 `runtime_scenario.ego_poses` 作为时间位姿来源

### 2.3 `backend/models.py`

新增 playback 相关模型：

- `PlaybackControlRequest`
- `PlaybackTickRequest`
- `PlaybackStateResponse`
- `PlaybackTickResponse`

---

## 3. API 与数据流（最新）

### 3.1 初始化流程

```text
Frontend  --POST /api/load-->  Backend
                               |- connect grpc(host:port)
                               |- Scenario(usdz_path)  # 本地解析 USDZ
                               |- PlaybackClock(scenario)
                               `- return {status, sequence_id}

Frontend  --GET /api/scenario--> Backend
                                 `- return cameras + time_range（来自 runtime scenario）
```

### 3.2 渲染流程

```text
Frontend  --POST /api/render(timestamp_us, camera_ids, scale)--> Backend
Backend:
  1) rig_matrix = scenario.ego_poses.interpolate_pose_matrix(timestamp_us)
  2) 对每个 camera:
       camera_matrix = rig_matrix @ T_sensor_rig
       组装 RGBRenderRequest
       grpc_stub.render_rgb(...)
  3) 返回 base64 JPEG 字典
```

### 3.3 播放控制流程（新增）

```text
Frontend --POST /api/playback/play--> Backend PlaybackClock.play()
Frontend --POST /api/playback/stop--> Backend PlaybackClock.stop()
Frontend --POST /api/playback/reset--> Backend PlaybackClock.reset()
Frontend --POST /api/playback/tick--> Backend PlaybackClock.tick(delta_us?)
Frontend --GET  /api/playback-------> Backend 当前播放状态
```

---

## 4. Playback API 说明

### 4.1 查询状态

`GET /api/playback`

返回字段：

- `is_playing`
- `speed`
- `current_time_us`
- `seconds_since_start`
- `done`

### 4.2 开始播放

`POST /api/playback/play`

请求体（可选）：

```json
{
  "speed": 1.0
}
```

### 4.3 暂停/重置

- `POST /api/playback/stop`
- `POST /api/playback/reset`

### 4.4 推进一帧

`POST /api/playback/tick`

请求体（可选）：

```json
{
  "delta_us": 100000
}
```

返回除 playback state 外，还包含：

- `new_track_ids`
- `removed_track_ids`

---

## 5. 前端对接建议（play / stop / tick）

### 5.1 推荐方式（以后端时钟为准）

前端控件语义建议映射为：

- 点击 Play：`POST /api/playback/play`
- 点击 Stop/Pause：`POST /api/playback/stop`
- 拖动时间轴或步进：`POST /api/playback/tick`（传 `delta_us`）
- 渲染帧：使用 `current_time_us` 调 `/api/render`

这种方式的好处：

- 时间推进统一在后端，避免前后端时钟漂移
- 后端可同时返回 track 激活变化（`new_track_ids`/`removed_track_ids`）
- 后续接入动态对象渲染时更容易对齐

### 5.2 兼容方式（前端本地时钟）

如果前端暂时仍用本地 `requestAnimationFrame` 维护 `currentTime_us`，也可继续工作：

- 直接用本地时间调用 `/api/render`
- 但建议逐步切到 `/api/playback/*`，确保状态单一来源

---

## 6. gRPC 直连调试（仍然有效）

当出现“backend 可连通但无图”时，建议绕过 backend，直接调用 `render_rgb` 做最小验证，快速区分：

- 参数拼装问题（backend）
- NuRec 渲染侧问题（容器/GPU）

> 说明：因为 backend 已改为本地解析 USDZ，gRPC 直连脚本不再要求先走 `get_available_scenes/cameras/trajectories` 作为主路径；重点是验证 `render_rgb` 是否健康。

---

## 7. 常见故障定位

### 7.1 gRPC 不可达 (`StatusCode.UNAVAILABLE`)

- 检查 NuRec 容器是否运行
- 检查端口映射 `46435:46435`
- 容器重启后等待 10~30 秒再试

### 7.2 渲染失败 (`NRenderer.render failed`)

- 常见为容器内渲染链路/GPU 资源问题
- 查看容器日志是否有 `cannot get cuda resource`、`JIT compiling failed`

### 7.3 OOM (`CUDA out of memory`)

- 降低 `resolution_scale`（如 `0.1` / `0.25`）
- 减少并发 camera 数量
- 重启容器释放显存
- 容器内可尝试 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## 8. 当前 REST <-> 后端实现映射

- `POST /api/load`
  - 建连 gRPC + 本地解析 USDZ + 初始化 playback
- `GET /api/scenario`
  - 返回本地解析得到的 camera/time_range
- `GET /api/trajectory`
  - 使用 `scenario.ego_poses` 插值采样
- `GET /api/pose`
  - 使用 `scenario.ego_poses` 时间插值
- `POST /api/render` / `GET /api/render/{camera_id}`
  - 调用 gRPC `render_rgb`
- `GET/POST /api/playback*`
  - 后端播放状态机控制与 tick 推进

