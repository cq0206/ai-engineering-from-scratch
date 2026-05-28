# 终端与 Shell

> 终端（terminal）是 AI 工程师的主战场。先在这里变得熟练起来。

**类型：** 学习
**语言：** --
**前置要求：** 第 0 阶段，第 01 课
**时长：** ~35 分钟

## 学习目标

- 使用管道（piping）、重定向（redirects）和 `grep` 从命令行过滤并处理训练日志
- 创建带有多个窗格的持久 tmux 会话，用于并行训练和 GPU 监控
- 使用 `htop`、`nvtop` 和 `nvidia-smi` 监控系统和 GPU 资源
- 使用 SSH、`scp` 和 `rsync` 在本地与远程机器之间传输文件

## 问题

你花在终端里的时间，会比花在任何编辑器里的时间都多。训练任务、GPU 监控、日志 tail、远程 SSH 会话、环境管理。每一条 AI 工作流都会碰到 shell。如果你在这里很慢，那你做什么都会慢。

这一课只讲 AI 工作真正需要的终端技能。不讲 Unix 历史。不深入 Bash 脚本。只讲你需要的内容。

## 概念

```mermaid
graph TD
    subgraph tmux["tmux 会话：训练"]
        subgraph top["顶部一行"]
            P1["窗格 1：训练任务<br/>python train.py<br/>Epoch 12/100 ..."]
            P2["窗格 2：GPU 监控<br/>watch -n1 nvidia-smi<br/>GPU: 78% | Mem: 14/24G"]
        end
        P3["窗格 3：日志与实验<br/>tail -f logs/train.log | grep loss"]
    end
```

三件事同时运行。一个终端。你可以分离会话，回家，重新 SSH 登录，再重新附着。训练会一直运行。

## 动手实践

### 第 1 步：了解你的 shell

检查你当前运行的是哪个 shell：

```bash
echo $SHELL
```

大多数系统使用 `bash` 或 `zsh`。两者都完全够用。本课程中的命令在这两种 shell 里都能运行。

你需要知道的关键点：

```bash
# Move around
cd ~/projects/ai-engineering-from-scratch
pwd
ls -la

# History search (most useful shortcut you'll learn)
# Ctrl+R then type part of a previous command
# Press Ctrl+R again to cycle through matches

# Clear terminal
clear   # or Ctrl+L

# Cancel a running command
# Ctrl+C

# Suspend a running command (resume with fg)
# Ctrl+Z
```

### 第 2 步：管道与重定向

管道会把命令连接起来。这就是你处理日志、过滤输出和串联工具的方式。你会一直用到它。

```bash
# Count how many times "loss" appears in a log
cat train.log | grep "loss" | wc -l

# Extract just the loss values from training output
grep "loss:" train.log | awk '{print $NF}' > losses.txt

# Watch a log file update in real time, filtering for errors
tail -f train.log | grep --line-buffered "ERROR"

# Sort experiments by final accuracy
grep "final_accuracy" results/*.log | sort -t= -k2 -n -r

# Redirect stdout and stderr to separate files
python train.py > output.log 2> errors.log

# Redirect both to the same file
python train.py > train_full.log 2>&1
```

你需要掌握的几个重定向符号：

| 符号 | 作用 |
|--------|-------------|
| `>` | 将 stdout 写入文件（覆盖） |
| `>>` | 将 stdout 追加到文件 |
| `2>` | 将 stderr 写入文件 |
| `2>&1` | 把 stderr 发送到与 stdout 相同的位置 |
| `\|` | 将一个命令的 stdout 作为下一个命令的 stdin |

### 第 3 步：后台进程

训练任务往往要跑几个小时。你不会想一直让终端开在那里。

```bash
# Run in background (output still goes to terminal)
python train.py &

# Run in background, immune to hangup (closing terminal won't kill it)
nohup python train.py > train.log 2>&1 &

# Check what's running in background
jobs
ps aux | grep train.py

# Bring a background job to foreground
fg %1

# Kill a background process
kill %1
# or find its PID and kill that
kill $(pgrep -f "train.py")
```

`&`、`nohup` 和 `screen`/`tmux` 的区别：

| 方法 | 关闭终端后仍能存活？ | 可以重新附着？ |
|--------|-------------------------|---------------|
| `command &` | 不可以 | 不可以 |
| `nohup command &` | 可以 | 不可以（查看日志文件） |
| `screen` / `tmux` | 可以 | 可以 |

任何会跑超过几分钟的任务，都用 tmux。

### 第 4 步：tmux

tmux 让你可以创建带多个窗格的持久终端会话。这是管理训练任务最有用的单个工具。

```bash
# Install
# macOS
brew install tmux
# Ubuntu
sudo apt install tmux

# Start a named session
tmux new -s training

# Split horizontally
# Ctrl+B then "

# Split vertically
# Ctrl+B then %

# Navigate between panes
# Ctrl+B then arrow keys

# Detach (session keeps running)
# Ctrl+B then d

# Reattach
tmux attach -t training

# List sessions
tmux ls

# Kill a session
tmux kill-session -t training
```

一个典型的 AI 工作流会话：

```bash
tmux new -s train

# Pane 1: start training
python train.py --epochs 100 --lr 1e-4

# Ctrl+B, " to split, then run GPU monitor
watch -n1 nvidia-smi

# Ctrl+B, % to split vertically, tail the logs
tail -f logs/experiment.log

# Now detach with Ctrl+B, d
# SSH out, go get coffee, come back
# tmux attach -t train
```

### 第 5 步：用 `htop` 和 `nvtop` 监控

```bash
# System processes (better than top)
htop

# GPU processes (if you have NVIDIA GPU)
# Install: sudo apt install nvtop (Ubuntu) or brew install nvtop (macOS)
nvtop

# Quick GPU check without nvtop
nvidia-smi

# Watch GPU usage update every second
watch -n1 nvidia-smi

# See which processes are using the GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

你会用到的 `htop` 快捷键：
- `F6` 或 `>`：按列排序（按内存排序可找到内存泄漏）
- `F5`：切换树状视图（查看子进程）
- `F9`：杀掉一个进程
- `/`：搜索进程名

### 第 6 步：远程 GPU 机器上的 SSH

当你租用云 GPU（Lambda、RunPod、Vast.ai）时，你会通过 SSH 连接。

```bash
# Basic connection
ssh user@gpu-box-ip

# With a specific key
ssh -i ~/.ssh/my_gpu_key user@gpu-box-ip

# Copy files to remote
scp model.pt user@gpu-box-ip:~/models/

# Copy files from remote
scp user@gpu-box-ip:~/results/metrics.json ./

# Sync a whole directory (faster for many files)
rsync -avz ./data/ user@gpu-box-ip:~/data/

# Port forward (access remote Jupyter/TensorBoard locally)
ssh -L 8888:localhost:8888 user@gpu-box-ip
# Now open localhost:8888 in your browser

# SSH config for convenience
# Add to ~/.ssh/config:
# Host gpu
#     HostName 192.168.1.100
#     User ubuntu
#     IdentityFile ~/.ssh/gpu_key
#
# Then just:
# ssh gpu
```

### 第 7 步：适用于 AI 工作的常用别名

把这些加入你的 `~/.bashrc` 或 `~/.zshrc`：

```bash
source phases/00-setup-and-tooling/10-terminal-and-shell/code/shell_aliases.sh
```

或者只复制你想要的那几条。关键别名如下：

```bash
# GPU status at a glance
alias gpu='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'

# Kill all Python training processes
alias killtraining='pkill -f "python.*train"'

# Quick virtual environment activate
alias ae='source .venv/bin/activate'

# Watch training loss
alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
```

完整列表见 `code/shell_aliases.sh`。

### 第 8 步：常见 AI 终端模式

这些模式会在实践里反复出现：

```bash
# Run training, log everything, notify when done
python train.py 2>&1 | tee train.log; echo "DONE" | mail -s "Training complete" you@email.com

# Compare two experiment logs side by side
diff <(grep "accuracy" exp1.log) <(grep "accuracy" exp2.log)

# Find the largest model files (clean up disk space)
find . -name "*.pt" -o -name "*.safetensors" | xargs du -h | sort -rh | head -20

# Download a model from Hugging Face
wget https://huggingface.co/model/resolve/main/model.safetensors

# Untar a dataset
tar xzf dataset.tar.gz -C ./data/

# Count lines in all Python files (see how big your project is)
find . -name "*.py" | xargs wc -l | tail -1

# Check disk space (training data fills disks fast)
df -h
du -sh ./data/*

# Environment variable check before training
env | grep -i cuda
env | grep -i torch
```

## 用起来

下面是本课程中各工具的典型使用时机：

| 工具 | 何时使用 |
|------|----------------|
| tmux | 每次训练任务都会用到（第 3 阶段及之后） |
| `tail -f` + `grep` | 监控训练日志 |
| `nohup` / `&` | 快速后台任务 |
| `htop` / `nvtop` | 调试训练变慢、OOM 错误 |
| SSH + `rsync` | 在云 GPU 上工作 |
| 管道 + 重定向 | 处理实验结果 |
| 别名 | 给重复命令省时间 |

## 练习

1. 安装 tmux，创建一个带三个窗格的会话，在其中分别运行 `htop`、`watch -n1 date` 和一个 Python 脚本。然后分离并重新附着。
2. 把 `code/shell_aliases.sh` 中的别名加入你的 shell 配置，并通过 `source ~/.zshrc`（或 `~/.bashrc`）重新加载。
3. 用 `for i in $(seq 1 100); do echo "epoch $i loss: $(echo "scale=4; 1/$i" | bc)"; sleep 0.1; done > fake_train.log` 创建一个假的训练日志，然后用 `grep`、`tail` 和 `awk` 只提取其中的 loss 值。
4. 为你能访问的一台服务器配置一个 SSH config 条目（或者用 `localhost` 练习语法）。

## 关键术语

| 术语 | 人们常说 | 真正含义 |
|------|----------------|----------------------|
| Shell | “终端” | 解释你输入命令的程序（bash、zsh、fish） |
| tmux | “终端多路复用器” | 让你在一个窗口里运行多个终端会话，并可分离/重新附着的程序 |
| Pipe | “那个竖线符号” | `\|` 运算符，把一个命令的输出作为另一个命令的输入 |
| PID | “进程 ID” | 分配给每个运行中进程的唯一编号，用于监控或终止进程 |
| nohup | “No hangup” | 让命令免受 hangup 信号影响，因此关闭终端也不会被杀掉 |
| SSH | “连到服务器上” | Secure Shell，一种可在远程机器上运行命令的加密协议 |

