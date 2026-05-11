# 安全带数据集下载与训练指南

> 适用于 3 号分工 · 行为识别算法 A 训练 `seatbelt.pt`
>
> 目标：得到一个能识别 `belt` / `no_belt`（或等价类别）的 YOLOv8 权重文件，
> 训练完成自动拷贝到 `behavior_algo_a/models/seatbelt.pt`，detector 启动即加载。

---

## 一、推荐数据集（按优先级排序）

### 🥇 1. Roboflow — seatbelt-detection by seatbelttraining（3489 张，推荐）
**链接：** https://universe.roboflow.com/seatbelttraining-7yh0f/seatbelt-detection-lb1ec

- 规模：3489 张已标注图
- 类别：`Belt` / `No Belt`
- 场景：真实驾驶舱角度
- 许可：CC BY 4.0
- 格式：原生 YOLO 可直接导出

### 🥈 2. Roboflow — seatbelt-detection by College（快速试用）
**链接：** https://universe.roboflow.com/college-704lm/seatbelt-detection-c7z5l

- 规模：较小（几百张）
- 类别：`Seatbelt`
- 适合快速 sanity check

### 🥉 3. Roboflow — seat-belt-detection by 2tech
**链接：** https://universe.roboflow.com/2tech/seat-belt-detection-udcfg

- 规模：中等
- 类别：`seat-belt` / `no-seat-belt`
- 代码中的默认 workspace/project 即此

### 备选 4. Kaggle 数据集（需 Kaggle 账号）
- https://www.kaggle.com/datasets/kvnpatel/seatbelt-detection-dataset
- 需要通过 kaggle CLI 下载，稍麻烦

---

## 二、两种下载路径

### ✅ 路径 A：Roboflow API（最稳，推荐）

**步骤：**

1. 注册 Roboflow 免费账号：https://roboflow.com/
2. 登录后 右上角头像 → **Settings** → **API**
3. 复制 **Private API Key**（形如 `xxxxxxxxxxxx`）
4. 安装依赖：
   ```bash
   pip install roboflow
   ```
5. 一键训练（下面选一个数据集即可）：

   ```bash
   # 数据集 #1 seatbelttraining (3489 张, 推荐)
   python scripts/train_seatbelt.py --source roboflow \
       --rf-api-key YOUR_KEY \
       --rf-workspace seatbelttraining-7yh0f \
       --rf-project seatbelt-detection-lb1ec \
       --rf-version 3 \
       --epochs 40 --imgsz 640 --device 0

   # 数据集 #3 2tech（默认）
   python scripts/train_seatbelt.py --source roboflow \
       --rf-api-key YOUR_KEY \
       --epochs 40
   ```

训练产物会自动拷贝到 `models/seatbelt.pt`。

### 🔧 路径 B：网页手动下载 Zip（无需 API key，但需登录）

**步骤：**

1. 打开上面任一数据集链接
2. 进入 → 右侧 **Download Dataset**
3. 选择 **Format: YOLOv8**（或 YOLOv5/v7 亦可）
4. 选择 **Download zip to computer** → 下载
5. 解压，会得到类似结构：
   ```
   seatbelt-detection-3/
   ├── data.yaml
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── test/ (可选)
   ```
6. 放到项目下：
   ```
   D:\Desktop\DMS\behavior_algo_a\data\seatbelt\
   ├── data.yaml
   ├── train/...
   └── valid/...
   ```
7. **改 data.yaml 路径为绝对路径**（关键）——打开 `data.yaml`，把：
   ```yaml
   train: ../train/images
   val: ../valid/images
   nc: 2
   names: ['Belt', 'No Belt']
   ```
   改为：
   ```yaml
   path: D:/Desktop/DMS/behavior_algo_a/data/seatbelt
   train: train/images
   val: valid/images
   nc: 2
   names: ['Belt', 'No Belt']
   ```
8. 训练：
   ```bash
   python scripts/train_seatbelt.py --source local \
       --data data/seatbelt/data.yaml \
       --epochs 40 --imgsz 640 --device 0
   ```

---

## 三、训练时长参考

| 数据集规模 | 设备 | Epochs | 预计耗时 |
|-----------|------|--------|---------|
| ~3000 张 | GPU (RTX 3060) | 40 | 15–25 分钟 |
| ~3000 张 | GPU (T4 Colab) | 40 | 25–40 分钟 |
| ~3000 张 | CPU (i5) | 40 | 4–8 小时 |
| ~1000 张 | GPU | 40 | 5–10 分钟 |
| ~1000 张 | CPU | 40 | 1–2 小时 |

**CPU 用户建议：** 在 Google Colab 免费 GPU 上训练，把 zip 上传后跑 train.py，再把 best.pt 下载回来放 `models/`。

---

## 四、训练完成检查

```bash
# 1. 确认权重存在
ls -la models/seatbelt.pt

# 2. 快速验证加载
python -c "from ultralytics import YOLO; m=YOLO('models/seatbelt.pt'); print(m.names)"
# 应输出类似 {0: 'Belt', 1: 'No Belt'}

# 3. 单图测试
python -c "
from ultralytics import YOLO
m = YOLO('models/seatbelt.pt')
r = m('data/seatbelt/valid/images/some_test.jpg')
r[0].show()
"

# 4. 跑实时客户端（自动加载）
python live_client.py --style monitor
```

Detector 启动日志应出现：
```
[Init] 加载安全带模型: models/seatbelt.pt
```

不再显示 "启用启发式"。

---

## 五、detector 的类别名识别规则

`behavior_detector.py` 中 `_judge_seatbelt()` 按类别名**关键字**判定"未系"：

```python
if "no" in name or "without" in name or "unbelted" in name:
    # 判定 no_seatbelt
```

所以数据集类别名只要**包含** `no`/`without`/`unbelted` 这几个词之一（大小写不敏感）就会被正确识别。常见可接受的命名：

- `No Belt` ✓
- `no_belt` ✓
- `No Seat Belt` ✓
- `without_seatbelt` ✓
- `unbelted` ✓

若你的数据集用了其他词（如 "empty"、"bare"），改 `_judge_seatbelt` 里的关键字即可。

---

## 六、常见问题

**Q: Roboflow 提示 "401 Unauthorized"？**
A: API key 错或没复制全。Settings → API 重新生成。

**Q: 训练中断能恢复吗？**
A: `python scripts/train.py ... --resume` 会从上次 epoch 恢复。

**Q: 显存不够（OOM）？**
A: `--batch 8` 或更小；或 `--imgsz 416`。

**Q: mAP@0.5 只有 0.3，怎么提升？**
A: ① 换更大数据集（数据集 #1 最好） ② `--epochs 80` ③ `--base yolov8s.pt`（比 n 大 3 倍）
