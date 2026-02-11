# NuRec Web Viewer - 设计文档

## 1. 项目概述

NuRec Web Viewer 是一个基于 Web 的神经重建场景查看器，允许用户通过浏览器查看 NuRec 渲染的多相机画面，并支持轨迹回放控制。

### 1.1 目标

- 提供 Web 界面查看 NuRec 渲染的图像
- 支持多相机同时显示（最多 3 个）
- 支持轨迹回放控制（播放/暂停/快进/快退）
- 独立于 CARLA 仿真器运行

### 1.2 核心依赖

- **NuRec Container**: 提供 gRPC 渲染服务
- **USDZ 文件**: 包含场景数据、轨迹、相机校准信息

---

## 2. 需求分析

### 2.1 功能需求

| ID | 需求 | 优先级 | 描述 |
|----|------|--------|------|
| FR-01 | 加载场景 | P0 | 加载 USDZ 路径并连接 NuRec gRPC，获取 scene/camera/trajectory 元数据 |
| FR-02 | 相机选择 | P0 | 显示可用相机列表，允许选择最多 3 个相机 |
| FR-03 | 图像渲染 | P0 | 调用 NuRec gRPC 服务渲染选中相机的图像 |
| FR-04 | 播放控制 | P0 | 支持 Play/Pause 切换 |
| FR-05 | 时间跳转 | P0 | 支持 ±1s, ±5s 快进/快退 |
| FR-06 | 时间轴滑块 | P1 | 支持拖动时间轴跳转到任意时间点 |
| FR-07 | 播放速度 | P2 | 支持 0.5x, 1x, 2x 播放速度 |
| FR-08 | 轨迹可视化 | P2 | 在地图上显示 ego 轨迹路径 |

### 2.2 非功能需求

| ID | 需求 | 描述 |
|----|------|------|
| NFR-01 | 响应时间 | 单帧渲染 + 传输 < 500ms |
| NFR-02 | 帧率 | 播放时目标 10-30 FPS（取决于渲染速度）|
| NFR-03 | 兼容性 | 支持 Chrome, Firefox, Safari 现代浏览器 |
| NFR-04 | 可扩展性 | 架构支持后续添加更多功能（如 LiDAR 渲染）|

### 2.3 约束条件

1. 浏览器不支持原生 gRPC，需要后端代理
2. NuRec 渲染是 GPU 密集型操作，需考虑性能
3. USDZ 文件需要预先加载到 NuRec Container

---

## 3. 系统架构

### 3.1 整体架构图

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              Frontend (React)                                  │
│                              Port: 3000                                        │
├────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                        Camera Display Grid (3 cameras)                    │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │  │
│  │  │  Camera 1       │  │  Camera 2       │  │  Camera 3       │           │  │
│  │  │  (front_wide)   │  │  (left_cross)   │  │  (right_cross)  │           │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘           │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                         Playback Controls                                 │  │
│  │  [<<] [<] [▶/⏸] [>] [>>]     ═══════════●═══════════════     00:05/02:30 │  │
│  │   -5s -1s Play  +1s +5s            Timeline Slider             time      │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Camera Selector: [x] front_wide  [ ] rear  [x] left_cross ...           │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ HTTP REST API
                                      ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                         Backend (Python FastAPI)                               │
│                         Port: 8000                                             │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────────┐   ┌──────────────────────┐   ┌───────────────────┐   │
│  │  Session Manager    │   │  Metadata Adapter    │   │  Render Proxy     │   │
│  │  - connect_grpc()   │   │  - list_cameras()    │   │  - render_cameras │   │
│  │  - load_scene()     │   │  - list_trajectory() │   │  - timeout guard  │   │
│  │  - select_scene()   │   │  - interpolate_pose  │   │  - error capture  │   │
│  └─────────────────────┘   └──────────────────────┘   └───────────────────┘   │
│                                                                                │
│                 直接调用 SensorsimService（无 scenario.py 依赖）                │
└────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ gRPC (SensorsimService)
                                      ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                         NuRec Container (Docker)                               │
│                         Port: 46435                                            │
├────────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │                     gRPC Server (SensorsimService)                      │   │
│  ├────────────────────────────────────────────────────────────────────────┤   │
│  │  get_available_scenes()    - 返回场景 ID 列表                           │   │
│  │  get_available_cameras()   - 返回相机列表和内参                          │   │
│  │  render_rgb()              - 渲染图像                                   │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │              Neural Reconstruction Engine (NRE)                         │   │
│  │  - 加载 USDZ 场景资产                                                    │   │
│  │  - 根据相机 pose 和 intrinsics 渲染                                      │   │
│  │  - 返回 JPEG 压缩图像                                                   │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 组件职责

#### 3.2.1 Frontend (React)

| 组件 | 职责 |
|------|------|
| `App.tsx` | 主应用入口，管理全局状态 |
| `CameraGrid.tsx` | 显示多相机画面网格 |
| `PlaybackControls.tsx` | 播放控制按钮和时间轴 |
| `CameraSelector.tsx` | 相机选择 checkbox 列表 |
| `useNuRecPlayback.ts` | 播放逻辑 hook |

#### 3.2.2 Backend (FastAPI)

| 模块 | 职责 |
|------|------|
| `main.py` | API 入口，gRPC 会话管理，渲染代理 |
| `models.py` | Pydantic 数据模型 |
| `proto/` | `common.proto`、`sensorsim.proto` 协议定义 |
| `backend/.cache/proto_generated` | 运行时自动生成的 Python gRPC stub（不入库） |

---

## 4. 数据流设计

### 4.1 初始化流程

```
Frontend                    Backend                      NuRec Container
    │                           │                           │
    │                           │                           │ (Container 已启动)
    │                           │                           │ (USDZ 已加载)
    │                           │                           │
    │── POST /api/load ─────────►│                           │
    │   { usdz_path, host, port }│                           │
    │                           │── gRPC get_available_scenes ───►│
    │                           │◄── scenes list ───────────│
    │                           │── gRPC get_available_cameras ──►│
    │                           │◄── cameras list ──────────│
    │                           │── gRPC get_available_trajectories ─►│
    │                           │◄── trajectory list ───────│
    │                           │                           │
    │◄── LoadResponse ─────────│                           │
    │                           │                           │
    │── GET /api/scenario ─────►│                           │
    │◄── ScenarioInfo ─────────│                           │
    │    - sequence_id          │                           │
    │    - time_range           │                           │
    │    - cameras[]            │                           │
    │                           │                           │
```

### 4.2 渲染流程

```
Frontend                    Backend                      NuRec Container
    │                           │                           │
    │── POST /api/render ──────►│                           │
    │   {                       │                           │
    │     timestamp_us: 12345,  │                           │
    │     camera_ids: [         │                           │
    │       "front_wide",       │                           │
    │       "left_cross"        │                           │
    │     ],                    │                           │
    │     resolution_scale: 0.25│                           │
    │   }                       │                           │
    │                           │                           │
    │                           │── interpolate ego_pose ──►│
    │                           │   at timestamp_us         │
    │                           │                           │
    │                           │── for each camera:        │
    │                           │   camera_pose =           │
    │                           │     ego_pose @ T_sensor_rig
    │                           │                           │
    │                           │── gRPC render_rgb ───────►│
    │                           │   RGBRenderRequest {      │
    │                           │     scene_id,             │
    │                           │     camera_intrinsics,    │
    │                           │     sensor_pose,          │
    │                           │     timestamp,            │
    │                           │     ...                   │
    │                           │   }                       │
    │                           │                           │── Neural Rendering
    │                           │◄── RGBRenderReturn ───────│   (GPU)
    │                           │    image_bytes (JPEG)     │
    │                           │                           │
    │◄── {                     │                           │
    │     timestamp_us,         │                           │
    │     images: {             │                           │
    │       "front_wide": "...",│ (base64)                  │
    │       "left_cross": "..." │                           │
    │     }                     │                           │
    │   }                       │                           │
    │                           │                           │
```

### 4.3 播放循环

```javascript
// 前端播放循环伪代码
while (isPlaying) {
    currentTime_us += timeStep_us * playbackSpeed;
    
    if (currentTime_us >= endTime_us) {
        isPlaying = false;
        break;
    }
    
    // 请求渲染
    images = await fetch('/api/render', {
        timestamp_us: currentTime_us,
        camera_ids: selectedCameras
    });
    
    // 更新显示
    updateCameraImages(images);
    
    // 帧率控制
    await sleep(1000 / targetFPS);
}
```

---

## 5. API 设计

### 5.1 REST API 接口

#### 5.1.1 加载场景

```
POST /api/load
```

**请求体**:
```json
{
    "usdz_path": "/path/to/scene.usdz",
    "nurec_host": "localhost",
    "nurec_port": 46435
}
```

**响应**:
```json
{
    "status": "loaded",
    "sequence_id": "clipgt-7f360cc2-371e-4606-9dc9-b9d0822928a8"
}
```

#### 5.1.2 获取场景信息

```
GET /api/scenario
```

**响应**:
```json
{
    "sequence_id": "clipgt-7f360cc2-371e-4606-9dc9-b9d0822928a8",
    "time_range": {
        "start_us": 1609459200000000,
        "end_us": 1609459350000000,
        "duration_seconds": 150.0
    },
    "cameras": [
        {
            "logical_name": "front_wide_120fov",
            "resolution": [1920, 1080],
            "T_sensor_rig": [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
        },
        {
            "logical_name": "left_cross",
            "resolution": [1920, 1080],
            "T_sensor_rig": [...]
        }
    ]
}
```

#### 5.1.3 获取轨迹

```
GET /api/trajectory?sample_interval_us=100000
```

**响应**:
```json
{
    "trajectory": [
        {
            "timestamp_us": 1609459200000000,
            "position": [100.0, 200.0, 0.5],
            "quaternion": [0, 0, 0, 1]
        },
        ...
    ]
}
```

#### 5.1.4 获取指定时间 Pose

```
GET /api/pose?timestamp_us=1609459200500000
```

**响应**:
```json
{
    "timestamp_us": 1609459200500000,
    "position": [100.5, 200.2, 0.5],
    "quaternion": [0, 0, 0.01, 0.9999]
}
```

#### 5.1.5 渲染图像

```
POST /api/render
```

**请求体**:
```json
{
    "timestamp_us": 1609459200500000,
    "camera_ids": ["front_wide_120fov", "left_cross", "right_cross"],
    "resolution_scale": 0.25
}
```

**响应**:
```json
{
    "timestamp_us": 1609459200500000,
    "images": {
        "front_wide_120fov": "/9j/4AAQSkZJRgABAQAAAQ...",  // base64 JPEG
        "left_cross": "/9j/4AAQSkZJRgABAQAAAQ...",
        "right_cross": "/9j/4AAQSkZJRgABAQAAAQ..."
    }
}
```

#### 5.1.6 渲染单个相机（直接返回图像）

```
GET /api/render/{camera_id}?timestamp_us=1609459200500000&scale=0.25
```

**响应**: `image/jpeg` 二进制数据

---

## 6. 关键数据结构

### 6.1 USDZ 文件内容

USDZ 是一个 ZIP 压缩包，包含以下 JSON 文件：

| 文件 | 内容 |
|------|------|
| `data_info.json` | 场景元数据：`sequence_id`, `pose-range` |
| `rig_trajectories.json` | Ego 轨迹、相机校准、世界坐标变换 |
| `sequence_tracks.json` | 所有动态物体的 track 数据 |
| `map.xodr` | OpenDRIVE 地图（可选） |

### 6.2 核心数据模型

```python
# Ego 轨迹数据结构 (来自 rig_trajectories.json)
{
    "rig_trajectories": [{
        "T_rig_worlds": [
            [[r11,r12,r13,tx], [r21,r22,r23,ty], [r31,r32,r33,tz], [0,0,0,1]],
            ...  # 4x4 变换矩阵列表
        ],
        "T_rig_world_timestamps_us": [1609459200000000, 1609459200033333, ...]
    }],
    "camera_calibrations": {
        "camera_0": {
            "logical_sensor_name": "front_wide_120fov",
            "T_sensor_rig": [[...], [...], [...], [...]],  # 4x4 矩阵
            "camera_model": {
                "type": "ftheta",
                "parameters": {
                    "resolution": [1920, 1080],
                    "principal_point": [960.0, 540.0],
                    "max_angle": 3.14159,
                    ...
                }
            }
        }
    },
    "T_world_base": [[...], [...], [...], [...]]  # 世界坐标变换
}
```

### 6.3 gRPC 消息结构

```protobuf
// RGBRenderRequest
message RGBRenderRequest {
    string scene_id = 1;
    uint32 resolution_h = 2;
    uint32 resolution_w = 3;
    CameraSpec camera_intrinsics = 4;
    fixed64 frame_start_us = 5;
    fixed64 frame_end_us = 6;
    PosePair sensor_pose = 7;
    repeated DynamicObject dynamic_objects = 8;
    ImageFormat image_format = 9;
    float image_quality = 10;
}

// Pose
message Pose {
    Vec3 vec = 1;   // translation [x, y, z]
    Quat quat = 2;  // rotation [w, x, y, z]
}
```

---

## 7. 相机 Pose 计算

### 7.1 坐标系说明

- **World 坐标系**: USDZ 文件中的全局坐标系
- **Rig 坐标系**: 车辆本体坐标系（Ego）
- **Sensor 坐标系**: 相机坐标系

### 7.2 变换链

```
T_camera_world = T_rig_world @ T_sensor_rig

其中:
- T_rig_world: Ego 车辆在世界坐标系中的位姿（4x4 矩阵）
- T_sensor_rig: 相机相对于车辆的固定变换（来自相机校准）
- T_camera_world: 相机在世界坐标系中的位姿（用于渲染请求）
```

### 7.3 Pose 插值

使用 `InterpolatedPoses` 类进行平滑插值：
- **平移**: 线性插值
- **旋转**: 球面线性插值 (SLERP)

```python
def interpolate_pose_matrix(self, timestamp: float) -> np.ndarray:
    # 找到前后两个关键帧
    start_pose, end_pose, t = self._get_interpolation_params(timestamp)
    
    # 平移线性插值
    interp_translation = start_translation + t * (end_translation - start_translation)
    
    # 旋转 SLERP 插值
    rotvec = (start_rotation.inv() * end_rotation).as_rotvec()
    slerp_result = start_rotation * Rotation.from_rotvec(rotvec * t)
    
    # 构建 4x4 矩阵
    result = np.eye(4)
    result[:3, :3] = slerp_result.as_matrix()
    result[:3, 3] = interp_translation
    return result
```

---

## 8. 性能优化

### 8.1 渲染优化

| 策略 | 描述 | 预期效果 |
|------|------|----------|
| 降低分辨率 | `resolution_scale=0.25` | 渲染时间减少 75% |
| 并行渲染 | 使用 `asyncio.gather` 并发渲染多相机 | 3 相机总时间接近单相机时间 |
| JPEG 压缩 | `image_quality=85` | 图像大小减少 50% |

### 8.2 网络优化

| 策略 | 描述 | 预期效果 |
|------|------|----------|
| 帧预取 | 播放时预渲染后续 2-3 帧 | 减少卡顿 |
| 帧缓存 | 缓存已渲染帧，拖动时间轴时复用 | 减少重复渲染 |
| WebSocket | 使用 WebSocket 推送图像 | 减少 HTTP 开销 |

### 8.3 前端优化

| 策略 | 描述 |
|------|------|
| 图像懒加载 | 只渲染可见相机 |
| 防抖 | 时间轴拖动时防抖请求 |
| 骨架屏 | 加载时显示占位符 |

---

## 9. 项目结构

```
/
├── DESIGN.md                    # 本设计文档
├── proto/
│   ├── common.proto             # gRPC common messages
│   └── sensorsim.proto          # gRPC sensorsim service/messages
├── backend/
│   ├── main.py                  # FastAPI 入口
│   ├── models.py                # Pydantic 数据模型
│   ├── requirements.txt         # Python 依赖
│   ├── .cache/
│   │   └── proto_generated/     # 运行时生成的 pb2 代码（不入库）
│   └── run.sh                   # 启动脚本
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # 主应用
│   │   ├── components/
│   │   │   ├── CameraGrid.tsx   # 相机网格
│   │   │   ├── PlaybackControls.tsx  # 播放控制
│   │   │   └── CameraSelector.tsx    # 相机选择
│   │   ├── hooks/
│   │   │   └── useNuRecPlayback.ts   # 播放逻辑
│   │   ├── types.ts             # TypeScript 类型
│   │   └── api.ts               # API 调用封装
│   ├── package.json
│   └── vite.config.ts
│
├── docker/
│   └── docker-compose.yml       # 启动 NuRec 容器
│
└── scripts/
    └── start_nurec.sh           # NuRec 容器启动脚本
```

---

## 10. 启动流程

### 10.1 前置条件

1. NuRec Docker 镜像已拉取
2. USDZ 场景文件已准备
3. GPU 可用

### 10.2 启动步骤

```bash
# 0. 登录并下载 USDZ
hf auth login
hf download nvidia/PhysicalAI-Autonomous-Vehicles-NuRec \
  "sample_set/25.07_release/Batch0001/026d6a39-bd8f-4175-bc61-fe50ed0403a3/026d6a39-bd8f-4175-bc61-fe50ed0403a3.usdz" \
  --repo-type dataset \
  --local-dir .

# 1. 启动 NuRec Container
./scripts/start_nurec.sh /path/to/scene.usdz

# 2. 启动 Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# 3. 启动 Frontend
cd frontend
npm install
npm run dev
```

### 10.3 访问

打开浏览器访问 `http://localhost:3000`

---

## 11. 后续扩展

### 11.1 功能扩展

- [ ] LiDAR 点云渲染和显示
- [ ] 3D 地图可视化（使用 Three.js）
- [ ] 动态物体标注显示
- [ ] 多场景切换
- [ ] 渲染参数调节（曝光、白平衡等）

### 11.2 性能扩展

- [ ] 服务端帧缓存 (Redis)
- [ ] 视频流输出 (WebRTC)
- [ ] GPU 多实例渲染

---

## 12. 当前实施进展（2026-02-11）

### 12.1 已完成修改

- 下载流程：文档和操作统一为 `hf download`（gated dataset 登录后下载）。
- `scripts/start_nurec.sh`：改为 Docker-only，删除 chroot/rootfs 相关逻辑。
- Backend：
  - 去除对 `scenario.py`/`track.py` 的强依赖。
  - 改为直接调用 `SensorsimService`（本地 `proto/` 生成 stub）。
  - `proto` 代码采用运行时生成到 `backend/.cache/proto_generated`。
  - 增加 gRPC 调用超时，避免接口长时间阻塞。

### 12.2 当前测试结果

- 服务启动：NuRec (`:46435`)、Backend (`:8000`)、Frontend (`:3000`) 均可启动。
- `/api/load`：已成功返回 `loaded`，可获取 scene id。
- 页面状态：前后端连接正常，可加载场景元数据。

### 12.3 已知问题

- 页面可能不显示渲染图片，NuRec 容器日志出现：
  - `NRenderer.render failed`
  - `cannot get cuda resource on the device 0`
- 该问题已定位到渲染/运行时侧（GPU 资源或容器 GPU 绑定），不属于前后端联通问题。
