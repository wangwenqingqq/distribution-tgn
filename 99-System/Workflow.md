# 使用与同步

本仓库遵循 research-workflow 0.1.0（上传包版本），一个课题对应一个私人 GitHub 仓库与一个 Obsidian vault。

## 本机工作流

1. Obsidian 常用目录保持在 main；助手用独立 work/<task> worktree。
2. 开始前阅读 AGENTS.md、README、课题卡片与来源表，检查 Git 状态并运行研究工作流 doctor。
3. 编辑中文笔记，更新索引与交接；真实代码、数据和权重留在权威位置。
4. 检查精确 diff，按具名文件运行 publish 预览，再 apply；不 force push。旧仓库另外运行本仓库 scripts/check.py。
5. 创建真实 PR，审阅后由获授权的人合并；本次对已有仓库只提交 PR，不代为合并。
6. 合并后，在干净 main 使用工作流 pull；未合并的新笔记不会出现在 main 的 Obsidian vault。需要审阅时使用 GitHub PR 或直接阅读任务工作树文件。

## 另一台设备

在新设备独立建立 GitHub 身份验证，克隆 main，以 Obsidian 正常的“打开文件夹为仓库”操作打开该课题。不要复制其他设备的令牌、SSH 私钥、助手配置或 .obsidian 设置；不自动修改 Obsidian 注册表。

## 本地证据

_local/Source-Map.md 仅作本机定位，默认忽略，不是加密存储。可移植笔记只保留逻辑来源、版本与哈希。遇到用户改动、冲突、扫描拒绝或远端分歧就停止发布并交接，不丢弃改动。
