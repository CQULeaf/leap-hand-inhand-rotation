# leap-hand-inhand-rotation

基于 Isaac Lab 的 LEAP Hand 手内圆柱体旋转强化学习框架。项目围绕 HORA 风格的两阶段策略学习展开：Stage1 使用仿真中的特权信息训练 teacher policy，Stage2 训练只依赖本体感知历史的 adaptation module，并提供 ID/OOD 评估、指标汇总和真实 LEAP Hand 部署入口。

这个仓库从作者毕设实验中独立整理出来，目标不是只保存代码，而是作为一个可以展示、复习和继续验证的完整实验工程。

## 项目亮点

- Isaac Lab DirectRLEnv 环境：LEAP Hand 手内操控圆柱体绕 z 轴持续旋转。
- HORA 风格两阶段训练：privileged teacher + proprioceptive student adaptation。
- 明确的仿真评估接口：支持 `fixed`、`id`、`ood` 三类 preset，并导出 `episodes.csv` 和 `summary.json`。
- 可复习的任务变体：注册了 NoObjPos、NoCom、NoDyn、PD randomization、observation noise、wide CoM 等环境变体。
- Sim2Real 部署入口：包含 stage2 policy 加载、sim-real 关节顺序映射、warmup grasp、Dynamixel 控制参数和 real trial 记录脚本。
- 脚本化工作流：训练、播放、评估、TensorBoard、评估汇总、抓取缓存生成都封装在 `scripts/` 下。

## 方法概览

```text
                 privileged state
       scale / mass / friction / CoM / object pose
                         |
                         v
Stage1 simulation   teacher latent + policy head   action
      policy obs -------------------------------> LEAP hand
   3-frame joint/target history

                         |
                         | imitation target
                         v
Stage2 adaptation  proprio history encoder  ->  student latent
                  30-frame joint/target history
                         |
                         v
                 same frozen policy head
```

Stage1 的 policy observation 是 96 维：`3 * (16 joint_pos + 16 action_target)`。默认 privileged critic/state 是 9 维：object position、scale、mass、friction、CoM。Stage2 不再依赖 privileged state，而是用 30 帧 proprio history 学出能够替代 teacher latent 的 student latent。

## 仓库结构

```text
scripts/
  train/                  # stage1 / stage2 训练入口
  eval/                   # play 与 quantitative evaluation
  deploy/                 # stage2 真机部署与真实试验记录
  tools/                  # task list、TensorBoard、grasp cache、评估汇总
  internal/               # 脚本调用的 Python 训练/播放实现
source/LEAP_Isaaclab/
  LEAP_Isaaclab/assets/   # LEAP Hand USD 与 cylinder URDF
  LEAP_Isaaclab/tasks/    # cylinder rotation task 注册、env cfg、DirectRLEnv
  LEAP_Isaaclab/utils/    # HORA adaptation、rl_games 组件、ADR 与观测工具
  LEAP_Isaaclab/deployment_scripts/
                           # stage2 policy 与 Dynamixel 控制器
docker/                   # Isaac Lab 容器构建入口
docs/                     # 部署审查等项目文档
```

核心文件：

- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_cylinder_rotation/__init__.py`：Gym task 注册。
- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_cylinder_rotation/leap_hand_env_cfg.py`：仿真、奖励、随机化和观测配置。
- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_cylinder_rotation/cylinder_rotation_env.py`：环境主逻辑。
- `source/LEAP_Isaaclab/LEAP_Isaaclab/utils/hora_adaptation.py`：Stage1/Stage2 策略加载、latent 计算、checkpoint 保存。
- `scripts/eval/evaluate_policy.py`：评估循环和指标导出。
- `source/LEAP_Isaaclab/LEAP_Isaaclab/deployment_scripts/cylinder_rotation_stage2.py`：真机 stage2 控制循环。

## 环境要求

已验证环境：

- Ubuntu 22.04
- NVIDIA GPU，已验证 RTX 5060 Laptop GPU 8GB
- Isaac Sim / Isaac Lab 可用
- Conda 环境：`env_isaaclab`
- Python 3.11 in Isaac Lab environment

脚本默认会尝试激活 `env_isaaclab`。如果当前 shell 没有 `conda` 命令，脚本会尝试加载 `/home/tools/anaconda3/etc/profile.d/conda.sh`。

### 资产要求

LEAP Hand 的 `.usd` 文件通过 Git LFS 管理。克隆仓库后请确认它们不是 LFS pointer：

```bash
git lfs install
git lfs pull
python3 scripts/tools/check_assets.py
file source/LEAP_Isaaclab/LEAP_Isaaclab/assets/leap_hand_v1_right/leap_hand_right.usd
head -n 3 source/LEAP_Isaaclab/LEAP_Isaaclab/assets/leap_hand_v1_right/leap_hand_right.usd
```

如果看到 `version https://git-lfs.github.com/spec/v1`，说明资产没有还原，Isaac Lab 会在创建 `Articulation` 时失败。正常文件应显示为 USD crate。

本仓库当前本机验证时已恢复并固化检查 4 个 LEAP Hand USD：

- `leap_hand_right.usd`
- `configuration/leap_hand_right_base.usd`
- `configuration/leap_hand_right_physics.usd`
- `configuration/leap_hand_right_sensor.usd`

`scripts/tools/check_assets.py` 会检查文件是否存在、是否仍为 Git LFS pointer、文件大小和 sha256。`.pre-commit-config.yaml` 已加入本地 hook `check-leap-assets`，用于在提交前提前发现 pointer 文件。

如果当前环境没有安装 `git lfs` 命令，仍可以先运行 `python3 scripts/tools/check_assets.py` 确认工作区中的 USD 是否为真实二进制文件。发布或推送前，请在具备 Git LFS 的环境中确认 4 个 USD 对象已经上传到远端 LFS 存储。

## 快速开始

所有命令默认从仓库根目录运行。当前脚本没有可执行位时，请用 `bash scripts/...` 调用。

### 1. 基础检查

```bash
git status --short
python3 scripts/tools/check_assets.py
find scripts -name '*.sh' -print0 | xargs -0 bash -n
python3 -m py_compile \
  scripts/tools/check_assets.py \
  scripts/tools/generate_grasp_cache.py \
  scripts/tools/summarize_evaluations.py \
  scripts/tools/list_tasks.py
```

### 2. 检查 Isaac Lab task 注册

```bash
source /home/tools/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH="$PWD/source/LEAP_Isaaclab:${PYTHONPATH:-}"
python scripts/tools/list_tasks.py
```

预期能看到 `Isaac-CylinderRotation-Leap` 以及多个 `Isaac-CylinderRotation-Leap-*` 任务变体。

## 训练流程

### Stage1：训练 privileged teacher policy

最小 smoke test：

```bash
bash scripts/train/stage1.sh \
  --profile debug \
  --num-envs 64 \
  --max-iterations 2 \
  --run-name smoke_stage1
```

常规本地训练入口：

```bash
bash scripts/train/stage1.sh --profile local-5060
```

云端或更大 GPU：

```bash
bash scripts/train/stage1.sh --profile cloud-4090
```

Stage1 输出默认位于：

```text
logs/rl_games/leap_hand_cylinder_rotation/<run-name>/
```

checkpoint 通常在：

```text
logs/rl_games/leap_hand_cylinder_rotation/<run-name>/nn/
```

### Stage2：训练 proprioceptive adaptation module

Stage2 需要一个 Stage1 checkpoint。最小 smoke test：

```bash
bash scripts/train/stage2.sh \
  --profile debug \
  --stage1-checkpoint /abs/path/to/stage1.pth \
  --num-envs 4 \
  --max-steps 16 \
  --save-every 16 \
  --log-every 8 \
  --run-name smoke_stage2
```

常规本地训练入口：

```bash
bash scripts/train/stage2.sh \
  --profile local-5060 \
  --stage1-checkpoint /abs/path/to/stage1.pth
```

Stage2 输出默认位于：

```text
logs/hora_stage2/leap_hand_cylinder_rotation/<run-name>/
```

关键 checkpoint：

```text
logs/hora_stage2/leap_hand_cylinder_rotation/<run-name>/nn/model_best.pt
logs/hora_stage2/leap_hand_cylinder_rotation/<run-name>/nn/model_last.pt
```

## 评估流程

### Stage1 fixed eval

```bash
bash scripts/eval/evaluate.sh \
  --policy-type stage1 \
  --stage1-checkpoint /abs/path/to/stage1.pth \
  --eval-preset fixed \
  --num-envs 32 \
  --num-episodes 32 \
  --run-name eval_stage1_fixed
```

### Stage2 ID/OOD eval

```bash
bash scripts/eval/evaluate.sh \
  --policy-type stage2 \
  --stage2-checkpoint /abs/path/to/model_best.pt \
  --eval-preset id \
  --num-envs 256 \
  --num-episodes 256 \
  --run-name eval_stage2_id
```

```bash
bash scripts/eval/evaluate.sh \
  --policy-type stage2 \
  --stage2-checkpoint /abs/path/to/model_best.pt \
  --eval-preset ood \
  --num-envs 256 \
  --num-episodes 256 \
  --run-name eval_stage2_ood
```

评估输出默认位于：

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

主要指标：

- `success_rate`：达到旋转阈值的 episode 比例。
- `ttf_mean`：time-to-fall 或 episode 持续时间均值。
- `net_rotation_turns_mean`：净旋转圈数。
- `mean_projected_angular_velocity_mean`：绕目标轴的平均角速度。
- `mean_object_linear_velocity_mean`：物体平动速度，越大通常越不稳定。
- `mean_command_torque_l1_mean`：命令扭矩尺度。
- `mean_latent_mse_mean`：Stage2 student latent 与 teacher latent 的差距。
- `mean_teacher_student_action_l2_mean`：Stage2 student action 与 teacher action 的差距。

## 论文主要结论

以下内容摘自作者毕设论文中的正式实验结论，README 先保留核心结论与指标，完整论文后续公开时可在这里补链接。

仿真评估显示，Stage2 的 Deploy-Refined policy 在 ID 场景保持 teacher 级别表现，并在 OOD 场景相对 Deploy-Base 提升了成功率、生存时间、净旋转圈数和轴向角速度。

| Policy | Preset | Success | Survival | Turns | Spin | LinVel cm/s | Torque |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher | ID | 1.0000 | 0.9983 | 3.6950 | 1.1628 | 3.4478 | 2.3407 |
| Deploy-Base | ID | 0.9922 | 0.9966 | 3.6309 | 1.1444 | 3.5378 | 2.3575 |
| Deploy-Refined | ID | 1.0000 | 0.9983 | 3.7081 | 1.1669 | 3.6543 | 2.3373 |
| Teacher | OOD | 0.9766 | 0.9913 | 3.4684 | 1.0956 | 3.9620 | 2.4103 |
| Deploy-Base | OOD | 0.8750 | 0.9492 | 3.0422 | 0.9803 | 4.4935 | 2.4433 |
| Deploy-Refined | OOD | 0.9141 | 0.9758 | 3.2558 | 1.0420 | 4.2818 | 2.4087 |

消融结果表明，teacher-action consistency constraint 是 OOD 提升的主要来源；加宽 CoM 随机化单独使用时成功率提升不明显，但与 action consistency 结合后扩大了高风险样本覆盖，OOD success 从 `0.8750` 提升到 `0.9141`，action gap 降至 `0.3053`。

真实 LEAP Hand 试验中，同一 Deploy-Refined policy 可在标准圆柱、小尺寸圆柱和高瓶状近圆柱物体上重复完成低速可控旋转。结论范围仍限定在圆柱参数泛化和小范围近圆柱迁移，不宣称任意物体通用操作。

## 当前验证状态

更新时间：2026-05-08。

本机已完成从任务注册到 Stage2 评估的 smoke test。结果只证明流程闭环，不代表最终策略性能，因为训练步数被刻意压到很小。

| 阶段 | 结果 | 产物 |
| --- | --- | --- |
| 任务注册 | 通过，12 个 LEAP task | `scripts/tools/list_tasks.py` |
| Stage1 train smoke | 通过，2 epochs | `logs/rl_games/leap_hand_cylinder_rotation/smoke_stage1_codex_assets_fixed_20260508_174754/nn/last_leap_hand_cylinder_rotation_ep_2_rew__-64.89644_.pth` |
| Stage1 fixed eval | 通过，4 episodes | `logs/evaluation/leap_hand_cylinder_rotation/smoke_stage1_eval_codex_20260508_175150/summary.json` |
| Stage2 train smoke | 通过，16 env steps | `logs/hora_stage2/leap_hand_cylinder_rotation/smoke_stage2_codex_20260508_175617/nn/model_best.pt` |
| Stage2 fixed eval | 通过，4 episodes | `logs/evaluation/leap_hand_cylinder_rotation/smoke_stage2_eval_codex_20260508_175751/summary.json` |
| LEAP USD asset check | 通过，4 个 USD 均为真实 USD crate | `scripts/tools/check_assets.py` |
| 真机部署只读审查 | 已完成，未连接硬件 | `docs/deployment_readiness_review.md` |

Smoke eval 摘要：

```text
stage1 fixed: success=0.0000, TTF=0.4183, Turns=0.0320
stage2 fixed: success=0.0000, TTF=0.4917, Turns=0.0679, LatentMSE=0.044541, ActionGap=0.017634
```

## 可视化与日志

Stage1 TensorBoard：

```bash
bash scripts/tools/tensorboard_stage1.sh
```

Stage2 TensorBoard：

```bash
bash scripts/tools/tensorboard_stage2.sh
```

如果端口冲突，可查看脚本参数或手动指定其他端口。

## 真机部署

本仓库已经完成真机部署前只读审查，记录在 `docs/deployment_readiness_review.md`。本次审查没有连接串口、没有执行 `--dry-run`、没有写电机参数。

真机部署入口位于：

```text
scripts/deploy/stage2.sh
scripts/deploy/real_trial.sh
source/LEAP_Isaaclab/LEAP_Isaaclab/deployment_scripts/cylinder_rotation_stage2.py
```

部署脚本会连接 Dynamixel 串口设备，并可能写电机控制模式、开扭矩、设置 P/D/current limit。执行前必须确认：

- LEAP Hand 固定安全，运动范围内无障碍物；
- 电源、电流限制、急停方案明确；
- 串口路径正确，例如 `/dev/ttyUSB0`；
- 使用的 stage2 checkpoint 明确；
- `--max-steps` 和 `--disable-torque-on-exit` 设置合理；
- 已决定是否记录真实试验表格和日志。

命令草案：

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

带真实试验元数据记录：

```bash
bash scripts/deploy/real_trial.sh \
  --object-id object01 \
  --object-name id_cylinder \
  --object-category ID \
  --checkpoint /abs/path/to/model_best.pt \
  --port /dev/ttyUSB0 \
  --max-steps 600
```

注意：`--dry-run` 仍会尝试连接并读取电机，只是跳过部分写命令，因此仍属于硬件相关操作。

正式部署前建议先处理审查中列出的高/中风险项：构造阶段异常时的 torque shutdown、配置前显式 disable torque、`--dry-run` 命名或离线 inference-only 模式、sim-real offset 常量统一、ID/映射只读核对。

## 常见问题

### `pxr` import 失败

不要在普通 Python 进程里直接 `import LEAP_Isaaclab.tasks` 来判断任务注册。Isaac/Omniverse 的 `pxr` 模块通常需要先通过 `AppLauncher` 初始化。使用：

```bash
python scripts/tools/list_tasks.py
```

### Isaac 打不开 `leap_hand_right.usd`

大概率是 Git LFS 文件没有还原。检查 `.usd` 文件是否为 LFS pointer，并重新拉取 LFS 资产。

### `num_envs` 与 rl_games batch 不兼容

Stage1 脚本会检查：

```text
num_envs * horizon_length % minibatch_size == 0
minibatch_size % seq_length == 0
```

默认 `horizon_length=96`、`minibatch_size=3072`、`seq_length=12`。debug 时可以使用 `--num-envs 64`、`--num-envs 256` 等能整除的配置。

### 评估没有打印 summary，但有退出码 0

先检查输出目录：

```bash
find logs/evaluation -name summary.json -o -name episodes.csv
```

有些 Isaac 日志会冲掉终端最后几行，但 `summary.json` 和 `episodes.csv` 已写出即可视为评估链路通过。

## 下一步建议

1. 提交前运行 `python3 scripts/tools/check_assets.py`，并确认 Git LFS 远端确实包含 4 个 USD 对象。
2. 基于 `docs/deployment_readiness_review.md` 做一轮部署安全小修，优先处理 torque shutdown 和配置前 disable torque。
3. 清理展示用实验目录：保留一组正式 Stage1、Stage2、ID/OOD eval、real trial 结果，smoke run 只作为开发记录。
4. 论文公开后，把论文 URL 和个人网站展示页 URL 补到 README；当前不在仓库内制作展示素材。
5. 真机恢复可用后，再按审查清单做一次低风险短步数部署试验，并同步更新真实试验表格。
