# MuscleMimic 本地肌骨 Demo

项目：`/Users/zsc/Downloads/musclemimic-web-demo-20261025`。

## 启动

```sh
cd /Users/zsc/Downloads/musclemimic-web-demo-20261025
export PATH="$PWD/.runtime/node/bin:$PATH"
npm run demo
```

打开 **http://127.0.0.1:4187/musclelab/**。已运行服务时直接访问，不要重复占用端口。开发模式可用 `npm run dev`，同样包含本地录制 API。默认用 WASM CPU，原始实验也可切换 WebGPU。

## 一键工作台

左侧选择输入，右侧固定显示结果；参数和诊断默认折叠。

- **摄像头**：点「一键启动」，授权后自动倒数 3 秒 → 录制 2 秒 → 生成 → 同步回放。过程中同一个按钮变为「取消」，取消会释放摄像头并停止本机任务。
- **示例视频**：来源选「示例视频」，点一次启动即可重新生成并自动播放。
- **本地视频**：拖入或选择视频后点启动，只分析前 2 秒。
- **回放**：一个播放／暂停／重播按钮，加可拖动的时间轴和倍速。原始实时仿真实验不能跳帧，生成的结果可以任意拖动；视频与三维状态一起定位。
- **历史**：结果下拉框切换，切换不会重跑处理。最近 6 条结果及对应视频保存在此浏览器的 IndexedDB，刷新后恢复最近一次结果；超出 6 条自动移除最旧缓存。结果右上角「···」可下载完整记录或删除当前浏览器缓存。
- **自动归档**：原视频、姿态点和物理结果自动保存到 `reports/recordings/<UUID>/`，分别为 `source.video`（保留原容器格式）、`poses.json`、`result.json`，时间与标题在 `metadata.json`。刷新页面会补存最近 6 条旧历史（旧历史没有原始姿态点，只能补存视频和轨迹）。浏览器删除/淘汰缓存不会删除本机归档；磁盘归档须自行管理，不会自动清除。无需再次手动导出。

输入与输出相互独立：生成下一段时，旧结果继续保留并可查看。失败或取消不会替换旧结果。界面保留整体误差、当前帧误差、根部误差和曲线；性能、种子、起点和后端收进「参数与诊断」。

仅在用户点启动后访问摄像头和开始录制。视频、姿态点和结果通过回环地址交给本机服务处理和保存，不发送到云端。关闭、取消或采集时隐藏页面会释放摄像头。浏览器缓存不可用时仍会尝试本机归档；归档服务失败会在诊断日志中提示，下次刷新重试。处理中取消或失败可能只留下姿态输入；完整视频在成功生成后保存。

没有足够的人体检测、严重遮挡、片段过短或解码失败会显示错误；本机处理超过 45 秒停止。取消等待中的摄像头权限也会立即恢复按钮，随后到达的设备流会被释放。

**流程已经可运行，单目新动作的跟踪精度仍是实验状态。** 不微调权重，比较现有 `checkpoint_13095` 与已发布的 `amathislab/mm-fullbody-base` 的完整物理仿真；前者已针对原始动作适配，并非对任意动作验证过的通用策略。单目深度、地面方向与根部位移是估计值；要求固定相机、直立起始、至少一只脚着地，不支持可靠的跳跃、快速动作或遮挡。画面一直展示真实误差，失败不会改为通过或替换成原始示例轨迹。

2026-10-26 同参考轨迹回归：两条真实用户录制中，前一条平均／末帧／根部最大误差从 143／183／201 mm 降到 93／115／109 mm，通过原有验收阈值；最后一条从 956／1571／1717 mm 降到 404／946／841 mm，**后半段仍倒地，未通过**。示例经过全新单目重建后平均约 234 mm，也未通过。完整统计见 `reports/tracking-fix/verification-summary.json`，真实录制的前后完整回放位于相应 UUID 目录的 `comparison.mp4`。浏览器自动测试使用 Chrome 的视频设备替身，不访问物理摄像头。

起始关节速度新增 100 ms 时间窗估计，缓解首帧差分尖峰。每段最多比较 7 条完整仿真，搜索预算为 12 秒（已开始的单次仿真完成后停止）；保留原始方案，候选必须在平均、末帧、根部最大误差和倒地帧数上均不劣于原始方案才可入选。目标轨迹与验收阈值不变，不拼接各条仿真的片段。倒地帧数是相对参考的根部下落 >0.35 m 或头部下落 >0.55 m 的诊断启发式，并非独立的生物力学判定。`result.json` 记录所有候选的指标、选中控制器及初始状态；旧历史保持原结果，新录制才使用新流程。

可选基础策略导出保存在 `inputs/mm-fullbody-base/onnx/`，不增加浏览器下载。其 ONNX 与原始 Flax actor 的对照最大误差为 1.55e-6，来源、校验和及验证值见同目录 `provenance.json`。服务验证该文件后才启用；缺失或校验失败则仅比较现有策略，错误记入结果。没有训练或更新任何策略权重。

## 计算与播放的边界

- 实时预览：MediaPipe Pose Landmarker Lite，单人 33 点；同一输入视频帧与推理结果一起显示。预览不是肌骨仿真。
- 录制处理：约 30 Hz 姿态采样 → 保留耦合关节的 IK → 支撑脚根部位移估计 → 100 Hz 目标轨迹 → 冻结 ONNX 策略候选 → MuJoCo 3.17 前向动力学，354 条肌肉、500 Hz 物理。
- 无策略训练或微调，无额外根部力，无中途状态投影/重置。地面修正仅作用于参考轨迹。
- 结果播放：本机先算完真实物理轨迹，浏览器再重建渲染几何。页面明确标为「录制结果回放」，不会冒充实时策略计算。
- 视频连续解码并校准到仿真时钟；HTTP Range 支持实际视频定位。最新录制回放测试最大时间差约 20 ms，完整结果见 `reports/recording-test.json`。

## 环境

Node 24 位于 `.runtime/node/`。前端依赖通过 `npm ci` 安装；模型、视频与预导出 ONNX 在 `public/`，无需运行训练或导出脚本。

本机录制服务使用 `.venv-physics/bin/python`（Python 3.12、MuJoCo 3.17），并从已有 `.venv-export/lib/python3.12/site-packages` 补充 SciPy 和 ONNX Runtime。该路径只在本仓库内使用，不修改系统 Python。迁移时可在 `.venv-physics` 中直接安装 `environment/recording-requirements.txt` 中的依赖。API 只监听 127.0.0.1，限制同源 JSON、单任务、输入大小与处理时间。

## 验证

```sh
export PATH="$PWD/.runtime/node/bin:$PATH"
npm test
.venv-physics/bin/python -m unittest discover -s tests -p 'test_recording_selection.py'
.venv-physics/bin/python scripts/verify-tracking-selection.py # 本机已归档的固定参考回归
npm run test:physics
# 另一个终端保持 npm run preview；浏览器测试需要本机 Chrome
npm run test:video-camera  # 同步、设备释放、权限拒绝；需要 ffmpeg
npm run test:recording     # 一键启动、倒数取消、拖动、历史恢复、上传、删除、权限异常
npm run test:browser -- wasm
```

工作台回归报告为 `reports/workbench-test.json`，截图为 `reports/workbench-result.png` / `workbench-mobile.png`；同步与原实验报告仍在 `reports/`。摄像头自动测试使用公开示例视频作为 Chrome fake camera，不访问物理摄像头。`scripts/process-recording.py` 可通过 stdin 接收 `{frames:[{timestamp,landmarks,worldLandmarks}]}`，stdout 返回 NDJSON 进度和结果。每帧时间戳单位毫秒。

## 原始视频实验

原有多视角视频与离线参考保留。在这个模式，浏览器从当前 MuJoCo 状态构造 2418 维观测，ONNX Runtime Web 实时计算动作，每周期执行 5 个 2 ms 物理步。只有标准正态噪声预先保存，动作和物理状态在线计算。可选 3 个动作起点、5 个随机种子，参考只写入独立显示数据。

此前原生与浏览器 WebGPU/WASM 均为 13/15，通过条件为世界坐标 17 点平均误差 <100 mm、末帧 <150 mm、根部最大 <200 mm；起点 0.10 s 的 seed 3、4 未通过。结果不代表新视频泛化。原生/WASM 3.17 同控制输入的 100 物理步 qpos 最大差约 1.9e-13；ONNX/JAX 动作均值差约 6.2e-6，5 个状态观测有对照测试。原始统计见 `public/validation.json`。

## 静态部署

`dist/` 可以静态部署原始实验和实时二维姿态预览。**录制肌骨处理需要本机 Node/Python 服务，单独部署静态文件不能运行这部分。** MediaPipe、MuJoCo、ORT 和模型由本地资源提供，无 CDN。没有发布本次改动到 GitHub 或网站。

模型、视频和依赖的来源与许可见 `public/ATTRIBUTION.html`、`public/licenses/`。限非商用研究演示；肌肉激活是模型输出，不是实测肌电或临床评估。

界面交互参考 Gradio 的 [Blocks / progress](https://gradio.app/docs/gradio/blocks) 与 [Video](https://gradio.app/docs/gradio/video) 输入输出模式。保留现有 Three.js / MuJoCo 渲染，不新增 Gradio 运行依赖。
