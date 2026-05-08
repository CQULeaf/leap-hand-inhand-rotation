# 真机部署前只读审查

更新时间：2026-05-08。

本审查只阅读部署脚本与控制代码，没有连接串口、没有执行 `--dry-run`、没有写电机参数，也没有读取真实电机状态。

## 审查范围

部署链路：

```text
scripts/deploy/stage2.sh
  -> source/LEAP_Isaaclab/LEAP_Isaaclab/deployment_scripts/cylinder_rotation_stage2.py
     -> stage2_policy.HoraStage2Policy
     -> utils/leap_hand_utils.dynamixel_client.DynamixelClient
```

真实试验记录链路：

```text
scripts/deploy/real_trial.sh
  -> scripts/deploy/stage2.sh
  -> evaluation/real/tables/*.csv
  -> evaluation/real/logs/*.log
```

## 关键行为确认

- 电机 ID 固定为 `0..15`，没有运行前的 ID 扫描或映射校验。
- 控制频率默认 `30 Hz`，动作增量为 `1.0 / 24.0`。
- sim-real 映射在 `construct_sim_to_real_transformation()` 中硬编码。
- 部署启动后会加载 Stage2 checkpoint，再连接 Dynamixel 串口。
- 非 `--dry-run` 模式会写 operating mode、开扭矩、写 P/D/current limit，再进入 warmup 和策略循环。
- `--dry-run` 只跳过 motor configuration writes 和 goal position writes，仍会连接串口并轮询电机位置，因此仍属于硬件相关操作。
- `--disable-torque-on-exit` 只在 `deploy()` 的 `finally` 中执行；如果构造 controller 过程中失败，可能到不了这个 `finally`。

## 风险与建议

| 等级 | 位置 | 发现 | 建议 |
| --- | --- | --- | --- |
| 高 | `cylinder_rotation_stage2.py` controller 生命周期 | 若 `_configure_motors()` 开扭矩后、进入 `deploy()` 前发生异常，`deploy()` 的 `finally` 不会执行。虽然 `DynamixelClient.disconnect()` 有关闭扭矩逻辑，但依赖退出清理不够直观。 | 部署前建议改成显式 controller lifecycle：构造失败也尝试 `set_torque_enabled(False)`，或实现 context manager。 |
| 中 | `_configure_motors()` | 会先写 operating mode，再开扭矩；如果上一次异常退出导致电机已处于 torque enabled，当前代码没有先 disable torque 再改模式/参数。 | 真机前建议在配置开始处显式关闭扭矩，确认写模式和增益后再开扭矩。 |
| 中 | `--dry-run` | 名称容易让人误以为完全离线，但实际仍会连接并读取真实电机。 | 文档中继续标记为硬件操作；后续可另加 `--inference-only`，只加载 checkpoint 并跑假观测。 |
| 中 | sim-real offset | 部署文件中 `LEAPsim_to_LEAPhand()` 使用 `3.14159`，`LEAPhand_to_LEAPsim()` 使用 `3.14`；工具文件中两者均为 `3.14159`。 | 统一为一个常量，减少部署和工具函数之间的微小偏差。 |
| 中 | 电机 ID 与关节映射 | 默认假设 16 个电机 ID、物理接线和 `sim_to_real_indices` 完全匹配。 | 真机前做只读 ID 扫描、人工核对每个关节方向和限位；通过后记录到 README 或试验记录。 |
| 中 | 参数范围 | Python 层没有限制 `hz/kp/kd/curr_lim/warmup_seconds/max_steps` 的范围。 | 真机前用保守参数，并在脚本层增加范围校验。 |
| 低 | `scripts/deploy/stage2.sh` checkpoint 选择 | 如果不传 `--checkpoint`，脚本会优先使用固定的 Deploy-Refined 路径，存在误用旧 checkpoint 的可能。 | 正式部署命令一律显式传 `--checkpoint`。 |
| 低 | `scripts/deploy/real_trial.sh` 调用方式 | 脚本内部用 `./scripts/deploy/stage2.sh`，若文件没有可执行位会失败；README 已建议使用 `bash scripts/...`。 | 后续可改为 `bash scripts/deploy/stage2.sh`，或给脚本补可执行位。 |
| 低 | `torch.load` | 只适合加载可信 checkpoint。 | 对外说明 checkpoint 必须来自本项目训练产物或可信来源。 |

## 真机前检查清单

- [ ] 确认 checkpoint 路径，正式部署不依赖自动查找。
- [ ] 确认串口路径，例如 `/dev/ttyUSB0`，只读确认设备存在，不改 udev 或用户组。
- [ ] 确认 16 个电机 ID、接线顺序、关节方向和 `sim_to_real_indices` 对应关系。
- [ ] 确认电源、电流限制、急停方式、周围空间和夹持物尺寸。
- [ ] 首次运行使用低风险参数：较低 `curr-lim`、较短 `max-steps`、明确 `--disable-torque-on-exit`。
- [ ] 部署前先手动摆到安全姿态，确认 warmup pose 不会发生机械干涉。
- [ ] 真实试验前准备记录表字段：物体 ID、尺寸、材料、质量、是否成功、持续时间、估计圈数、视频文件。

## 命令草案

这些命令尚未在本次审查中执行。

```bash
bash scripts/deploy/stage2.sh \
  --checkpoint /abs/path/to/model_best.pt \
  --port /dev/ttyUSB0 \
  --hz 30 \
  --kp 600 \
  --kd 150 \
  --curr-lim 400 \
  --max-steps 600 \
  --disable-torque-on-exit
```

```bash
bash scripts/deploy/real_trial.sh \
  --object-id object01 \
  --object-name id_cylinder \
  --object-category ID \
  --checkpoint /abs/path/to/model_best.pt \
  --port /dev/ttyUSB0 \
  --max-steps 600
```
