# 基于深度强化学习的 LEAP Hand 手内旋转系统

本项目实现了 HORA 风格的两阶段策略学习流程在 LEAP Hand 低成本灵巧手上的应用与增强：先训练带特权信息的仿真 teacher policy，再训练只依赖本体感知历史的可部署 student policy。项目打通了“任务建模——仿真训练——实机部署”整个流程。本仓库开源所有相关代码，可用于仿真实验复现、定量评估和真实硬件部署。

视频展示：[[Website](https://yexuhang.com/projects/)]
论文展示：[[Paper](https://github.com/CQULeaf/leap-hand-inhand-rotation/blob/main/paper/main.pdf)]

## 项目亮点

1. 面向低成本 LEAP Hand，在无视觉、无触觉反馈条件下完成圆柱体连续手内旋转的仿真训练、统一评测和真实硬件闭环部署。
2. 将 HORA/RMA 类两阶段适应思路改造为适用于 LEAP 平台的 `8` 维技能先验：训练期使用对象位置、尺度、质量、摩擦和质心偏移等特权信息，部署期只依赖关节本体历史恢复同一先验。
3. 针对部署策略在 OOD 圆柱参数下的退化，引入 teacher-student action consistency，并扩大质心偏移覆盖范围；论文实验中 Deploy-Refined 将 OOD success 从 `0.8750` 提升到 `0.9141`，存活时间、净旋转圈数和轴向角速度同步改善。
4. 评测链路覆盖 Teacher、Deploy-Base、Deploy-Refined 三类策略，包含 ID/OOD 对比、特权信息消融、历史长度消融、编码器结构消融和动作一致性权重分析。
5. 同一 Deploy-Refined 策略可在真实 LEAP Hand 上闭环运行，并在标准圆柱、尺寸变化圆柱和细高近圆柱物体上产生可重复的低速受控旋转。结论范围限定在圆柱参数泛化和少量近圆柱对象迁移。

## 仓库结构

```text
pretrained/                 # 提供训练好的 Stage1 / Stage2 checkpoint
scripts/
  train/                    # Stage1 与 Stage2 训练入口
  eval/                     # 策略播放与定量评估
  deploy/                   # 真实 LEAP Hand 部署包装脚本
  tools/                    # 资产检查、任务列表、TensorBoard、指标汇总
  internal/                 # shell 脚本调用的 Python 实现
source/LEAP_Isaaclab/
  LEAP_Isaaclab/assets/     # LEAP Hand USD 与 cylinder URDF
  LEAP_Isaaclab/tasks/      # 任务注册、配置和 DirectRLEnv 实现
  LEAP_Isaaclab/utils/      # HORA adaptation、rl_games 工具、观测处理
  LEAP_Isaaclab/deployment_scripts/
                            # Stage2 policy wrapper 与 Dynamixel 控制器
docker/                     # Isaac Lab 容器入口
paper/                      # 论文以及相应的LaTeX源码
```

## 环境要求

项目所用环境：

- Ubuntu 22.04 LTS
- NVIDIA GeForce RTX 5060 Laptop GPU
- Isaac Sim `5.1.0`
- Isaac Lab `2.3.0`
- Conda 环境名：`env_isaaclab`
- Isaac Lab 环境内 Python `3.11.15`

脚本默认会尝试激活 `env_isaaclab`。如果当前 shell 没有 `conda` 命令，脚本会尝试加载 `/home/tools/anaconda3/etc/profile.d/conda.sh`。

## 资产与 Checkpoint

LEAP Hand USD 资产和预训练 checkpoint 通过 Git LFS 管理。克隆仓库后运行：

```bash
git lfs install
git lfs pull
python3 scripts/tools/check_assets.py
```

`scripts/tools/check_assets.py` 会检查 4 个必要 LEAP Hand USD 文件是否为真实 USD 文件，而不是 Git LFS pointer stub。

预训练权重：

```text
pretrained/stage1_teacher.pth
pretrained/stage2_deploy_refined.pt
```

如果未显式传入 checkpoint，`play_stage2.sh`、`evaluate.sh` 和 `deploy/stage2.sh` 会默认使用 `pretrained/stage2_deploy_refined.pt`。

## 快速开始

所有命令默认从仓库根目录运行。当前 `.sh` 文件不要求有可执行位，建议统一用 `bash scripts/...` 调用。

```bash
python3 scripts/tools/check_assets.py
find scripts -name '*.sh' -print0 | xargs -0 bash -n
```

列出已注册任务：

```bash
source /home/tools/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH="$PWD/source/LEAP_Isaaclab:${PYTHONPATH:-}"
python scripts/tools/list_tasks.py
```

在仿真中播放预训练 Stage2 策略：

```bash
bash scripts/eval/play_stage2.sh --fixed-eval
```

评估预训练 Stage2 策略：

```bash
bash scripts/eval/evaluate.sh \
  --policy-type stage2 \
  --eval-preset id \
  --num-envs 256 \
  --num-episodes 256 \
  --run-name eval_stage2_id
```

OOD 评估：

```bash
bash scripts/eval/evaluate.sh \
  --policy-type stage2 \
  --eval-preset ood \
  --num-envs 256 \
  --num-episodes 256 \
  --run-name eval_stage2_ood
```

评估结果默认写入：

```text
logs/evaluation/leap_hand_cylinder_rotation/<run-name>/
  episodes.csv
  summary.json
```

汇总多个评估结果：

```bash
python3 scripts/tools/summarize_evaluations.py \
  $(find logs/evaluation -name summary.json | sort) \
  --format markdown
```

## 训练

Stage1 本地训练入口：

```bash
bash scripts/train/stage1.sh --profile local-5060
```

Stage2 本地训练入口：

```bash
bash scripts/train/stage2.sh \
  --profile local-5060 \
  --action-loss-weight 1.0
```

训练输出：

```text
logs/rl_games/leap_hand_cylinder_rotation/<stage1-run>/
logs/hora_stage2/leap_hand_cylinder_rotation/<stage2-run>/
```

## 评估指标

评估脚本主要输出：

- `success_rate`：成功 episode 比例。
- `ttf_mean`：平均存活时间 / time-to-fall。
- `net_rotation_turns_mean`：圆柱净旋转圈数。
- `mean_projected_angular_velocity_mean`：绕目标轴平均旋转角速度。
- `mean_object_linear_velocity_mean`：物体平均线速度，可作为稳定性参考。
- `mean_command_torque_l1_mean`：命令扭矩尺度。
- `mean_latent_mse_mean`：Stage2 student latent 与 teacher latent 的误差。
- `mean_teacher_student_action_l2_mean`：Stage2 action 与 teacher action 的差距。

论文中的实验结果：

| Policy | Preset | Success | Survival | Turns | Spin | LinVel cm/s | Torque |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher | ID | 1.0000 | 0.9983 | 3.6950 | 1.1628 | 3.4478 | 2.3407 |
| Deploy-Base | ID | 0.9922 | 0.9966 | 3.6309 | 1.1444 | 3.5378 | 2.3575 |
| Deploy-Refined | ID | 1.0000 | 0.9983 | 3.7081 | 1.1669 | 3.6543 | 2.3373 |
| Teacher | OOD | 0.9766 | 0.9913 | 3.4684 | 1.0956 | 3.9620 | 2.4103 |
| Deploy-Base | OOD | 0.8750 | 0.9492 | 3.0422 | 0.9803 | 4.4935 | 2.4433 |
| Deploy-Refined | OOD | 0.9141 | 0.9758 | 3.2558 | 1.0420 | 4.2818 | 2.4087 |

Deploy-Refined 结合了更宽的 CoM 覆盖和 teacher-action consistency。在 OOD 设置中，它相对 Deploy-Base 将成功率从 `0.8750` 提升到 `0.9141`。

## 真实硬件部署

仓库提供部署脚本，但真实硬件执行必须单独按安全关键步骤处理。部署路径会连接 Dynamixel 电机，并可能设置控制模式、扭矩状态、P/D gain 和电流限制。

入口：

```text
scripts/deploy/stage2.sh
scripts/deploy/real_trial.sh
source/LEAP_Isaaclab/LEAP_Isaaclab/deployment_scripts/cylinder_rotation_stage2.py
```

运行前请确认 LEAP Hand 固定、电源与电流限制、急停方案、串口、checkpoint 和步数上限。

命令模板：

```bash
bash scripts/deploy/stage2.sh \
  --checkpoint pretrained/stage2_deploy_refined.pt \
  --port /dev/ttyUSB0 \
  --hz 30 \
  --kp 600 \
  --kd 150 \
  --curr-lim 400 \
  --max-steps 600 \
  --disable-torque-on-exit
```

记录真实试验表格和终端日志：

```bash
bash scripts/deploy/real_trial.sh \
  --object-id object01 \
  --object-name id_cylinder \
  --object-category ID \
  --checkpoint pretrained/stage2_deploy_refined.pt \
  --port /dev/ttyUSB0 \
  --max-steps 600
```

注意：`--dry-run` 仍会打开串口并读取电机状态，它不是离线 inference-only 模式。
