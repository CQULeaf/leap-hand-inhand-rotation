# 基于深度强化学习的 LEAP Hand 手内旋转系统

基于 Isaac Lab 的 LEAP Hand 手内圆柱旋转强化学习项目。仓库实现了 HORA 风格的两阶段策略学习流程：先训练带特权信息的仿真 teacher policy，再训练只依赖本体感知历史的可部署 student policy。

本仓库面向仿真实验复现、定量评估和谨慎的真实硬件部署整理。

## 项目亮点

- 基于 Isaac Lab `DirectRLEnv` 的 LEAP Hand 圆柱旋转任务。
- HORA 风格两阶段学习：privileged teacher + proprioceptive adaptation。
- 提供预训练 Stage1 / Stage2 checkpoint，可直接播放和评估。
- 提供 `fixed`、`id`、`ood` 评估 preset，并导出 `episodes.csv` 与 `summary.json`。
- 注册了 object pose、CoM、dynamics、PD randomization、observation noise、wide CoM 等任务变体，便于消融实验。
- 训练、评估、播放、TensorBoard、资产检查和部署入口都封装在 `scripts/` 下。

## 方法概览

```text
Stage1 teacher
  policy obs: 3 帧 joint/action-target 历史
  privileged state: object pose, scale, mass, friction, CoM
  output: action + teacher latent

Stage2 student
  input: 30 帧本体感知 joint/action-target 历史
  target: teacher latent，可选 teacher action consistency
  output: 不依赖 privileged state 的部署策略
```

默认维度：

- Stage1 policy observation：`96 = 3 * (16 joint_pos + 16 action_target)`
- Privileged state：`9 = object_pos(3) + scale + mass + friction + CoM(3)`
- Stage2 proprioceptive history：`30 * 32`
- Action space：`16` 个 LEAP Hand 关节

## 仓库结构

```text
pretrained/                 # 发布用 Stage1 / Stage2 checkpoint
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
docker/                     # 可选 Isaac Lab 容器入口
```

核心文件：

- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_cylinder_rotation/leap_hand_env_cfg.py`
- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_cylinder_rotation/cylinder_rotation_env.py`
- `source/LEAP_Isaaclab/LEAP_Isaaclab/utils/hora_adaptation.py`
- `scripts/eval/evaluate_policy.py`
- `source/LEAP_Isaaclab/LEAP_Isaaclab/deployment_scripts/cylinder_rotation_stage2.py`

## 环境要求

本地验证环境：

- Ubuntu 22.04
- NVIDIA GPU
- Isaac Sim / Isaac Lab 可用环境
- Conda 环境名：`env_isaaclab`
- Isaac Lab 环境内 Python 3.11

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

Stage1 teacher smoke test：

```bash
bash scripts/train/stage1.sh \
  --profile debug \
  --num-envs 64 \
  --max-iterations 2 \
  --run-name smoke_stage1
```

Stage1 本地训练入口：

```bash
bash scripts/train/stage1.sh --profile local-5060
```

Stage2 student smoke test。默认使用 `pretrained/stage1_teacher.pth` 作为 Stage1 teacher：

```bash
bash scripts/train/stage2.sh \
  --profile debug \
  --num-envs 4 \
  --max-steps 16 \
  --save-every 16 \
  --log-every 8 \
  --run-name smoke_stage2
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

代表性论文实验结果：

| Policy | Preset | Success | Survival | Turns | Spin | LinVel cm/s | Torque |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher | ID | 1.0000 | 0.9983 | 3.6950 | 1.1628 | 3.4478 | 2.3407 |
| Deploy-Base | ID | 0.9922 | 0.9966 | 3.6309 | 1.1444 | 3.5378 | 2.3575 |
| Deploy-Refined | ID | 1.0000 | 0.9983 | 3.7081 | 1.1669 | 3.6543 | 2.3373 |
| Teacher | OOD | 0.9766 | 0.9913 | 3.4684 | 1.0956 | 3.9620 | 2.4103 |
| Deploy-Base | OOD | 0.8750 | 0.9492 | 3.0422 | 0.9803 | 4.4935 | 2.4433 |
| Deploy-Refined | OOD | 0.9141 | 0.9758 | 3.2558 | 1.0420 | 4.2818 | 2.4087 |

Deploy-Refined 结合了更宽的 CoM 覆盖和 teacher-action consistency。在报告的 OOD 设置中，它相对 Deploy-Base 将成功率从 `0.8750` 提升到 `0.9141`。

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

## 开发检查

```bash
python3 scripts/tools/check_assets.py
find scripts -name '*.sh' -print0 | xargs -0 bash -n
python3 -m py_compile \
  scripts/tools/check_assets.py \
  scripts/tools/generate_grasp_cache.py \
  scripts/tools/summarize_evaluations.py \
  scripts/tools/list_tasks.py \
  scripts/eval/evaluate_policy.py \
  scripts/internal/hora_play_stage2.py \
  scripts/internal/hora_train_stage2.py \
  scripts/internal/rl_games_play.py \
  scripts/internal/rl_games_train.py
```

TensorBoard：

```bash
bash scripts/tools/tensorboard_stage1.sh
bash scripts/tools/tensorboard_stage2.sh
```

## 当前状态

- 仿真任务、训练脚本、评估脚本和资产检查脚本已整理完成。
- 预训练 Stage1 / Stage2 checkpoint 已放入 `pretrained/`。
- 真实硬件部署入口已保留，但执行前必须做明确的硬件安全检查。
- 论文和项目主页公开后，可在 README 中补充链接。
