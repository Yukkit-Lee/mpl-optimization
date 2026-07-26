# MTT 本地版本封存

当前基线为 `main@50dccea47ba48b9cfb3b9c59375d7da92e4e603b`，上游是
`https://github.com/GeorgeCazenavette/mtt-distillation.git`。当前 tracked
diff 仅包含 executable bit 变化；本地未跟踪研究源码与大数据/权重分开封存。

该目录不是公开仓库的简单引用，而是当前服务器真实 MTT 工作区的恢复材料。

- `commit.txt`：基线 commit。
- `branch.txt` / `remotes.txt`：原分支和远端。
- `mtt-history.bundle`：完整 Git 历史，可在离线环境中克隆。
- `mtt-working-tree.patch`：tracked 文件的二进制完整 patch；当前只包含文件模式变化。
- `mtt-untracked-source.tar.gz`：当前筛选出的未跟踪源码/配置/说明文件，不包含数据、权重、缓存和运行日志。
- `untracked-all.txt`：所有未跟踪文件盘点，便于回到旧服务器补取历史资产。
- `status.txt` / `diff-stat.txt`：封存时工作区状态。
- `SHA256SUMS`：三项核心恢复材料的校验值。

恢复时运行：

```bash
export PROJECT_ROOT=/root/autodl-tmp/mpl-optimization
export MTT_ROOT=/root/autodl-tmp/mtt-distillation
bash "$PROJECT_ROOT/scripts/restore_mtt_snapshot.sh"
```

大型 `.pt/.pth`、expert buffers、data、logged files 和 wandb 没有进入该源码快照，位置见
`migration/artifact-manifest.md`。
