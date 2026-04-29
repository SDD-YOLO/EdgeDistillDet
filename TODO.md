# 模型蒸馏全流程管理平台 - 开发任务清单

> 目标：构建一个支持本地训练、云端提交、实验管理和智能分析的模型蒸馏平台。

[x]补全断点续练的不可以切换设备和配置的参数的逻辑
[x]更改logo，并且让logo支持明暗切换
[x]将task: 任务类型 (detect/segment)mode: 运行模式 (train/val/predict)model: 学生模型权重路径data: 数据集 yaml 路径epochs: 总训练轮数time: 限时训练时长patience: 早停轮数batch: 批次大小imgsz: 图像尺寸save: 是否保存模型save_period: 间隔轮数保存cache: 数据集缓存 (false/ram/disk)device: 运行设备 (cpu/0/1)workers: 数据加载线程project: 结果保存根目录name: 实验文件夹名exist_ok: 覆盖同名文件夹pretrained: 是否使用预训练权重optimizer: 优化器类型 (auto/SGD/Adam)verbose: 输出详细日志seed: 随机种子deterministic: 固定随机因子single_cls: 是否单类别训练rect: 矩形训练cos_lr: 余弦学习率close_mosaic: 关闭马赛克增强轮数resume: 断点续训amp: 混合精度训练fraction: 数据集使用比例profile: 性能分析freeze: 冻结模型层数multi_scale: 多尺度训练compile: 模型编译加速overlap_mask: 掩码重叠mask_ratio: 掩码下采样dropout: 随机失活val: 训练时验证split: 验证集划分save_json: 保存验证 jsonconf: 置信度阈值iou: NMS 阈值max_det: 单图最大检测数half: 半精度dnn: OpenCV 推理plots: 生成训练图表end2end: 端到端模式source: 推理数据源vid_stride: 视频步长stream_buffer: 流缓冲visualize: 特征可视化augment: 推理增强agnostic_nms: 类别无关 NMSclasses: 指定检测类别retina_masks: 高分辨率掩码embed: 特征导出show: 实时预览save_frames: 保存帧save_txt: 保存标签save_conf: 保存置信度save_crop: 保存裁剪结果show_labels: 显示标签show_conf: 显示置信度show_boxes: 显示检测框line_width: 框线宽度format: 模型导出格式keras: Keras 导出optimize: 模型优化int8: INT8 量化dynamic: 动态输入simplify: 简化模型opset: ONNX 版本workspace: 工作空间nms: 启用 NMSlr0: 初始学习率lrf: 最终学习率系数momentum: 优化器动量weight_decay: 权重衰减warmup_epochs: 预热轮数warmup_momentum: 预热动量warmup_bias_lr: 偏置预热学习率box: 检测框损失权重cls: 分类损失权重cls_pw: 分类损失平衡系数dfl: 分布焦点损失权重pose: 姿态损失权重kobj: 关键点损失权重rle: 掩码编码系数angle: 旋转角度系数nbs: 标准化批次hsv_h: 色相增强hsv_s: 饱和度增强hsv_v: 亮度增强degrees: 旋转角度translate: 平移幅度scale: 缩放幅度shear: 错切变换perspective: 透视变换flipud: 上下翻转概率fliplr: 左右翻转概率bgr: BGR 通道转换mosaic: 马赛克增强mixup: 混合增强cutmix: 剪切混合copy_paste: 复制粘贴增强copy_paste_mode: 增强模式auto_augment: 自动增强策略erasing: 随机擦除cfg: 模型配置文件tracker: 追踪器配置save_dir: 自动生成的保存路径teacher_weights: 教师模型权重路径 teacher_cfg: 教师模型配置文件distill_mode: 蒸馏模式 (adaptive 自适应 /feature 特征 /response 响应)
temperature: 蒸馏温度 (推荐 2~8)alpha: 软标签损失权重 (推荐 0.3~0.8)beta: 特征蒸馏损失权重 (推荐 0.001~0.01)gamma: 关系蒸馏损失权重cls_distill_weight: 分类分支蒸馏权重box_distill_weight: 检测框分支蒸馏权重obj_distill_weight: 置信度分支蒸馏权重distill_loss_type: 损失类型 (kl/mse/l1) freeze_teacher: 冻结教师模型 (True/False)teacher_device: 教师模型运行设备distill_start_epoch: 蒸馏开始轮数distill_end_epoch: 蒸馏结束轮数dynamic_alpha: 动态权重调度开关dynamic_temperature: 动态温度调度开关 feature_layers: 特征蒸馏层选择lambda_kd: 总蒸馏损失系数use_adaptive_loss: 是否启用自适应损失

这些可以修改的参数直接全部在前端添加一个新的“高级参数配置”展示出来，确保所有的前端的参数设置窗口和后端是正确的映射关系
[x]下载依赖的CUDA版本的时候先查看一下设备的配置并下载对应的CUDA版本，确保下载的CUDA版本和设备配置是匹配的
[x]仔细检查代码中有没有硬编码，如果有的话需要修改
[x]仔细检查代码中有没有重复的代码，如果有的话需要合并
[x]仔细检查代码中有没有潜在的bug，如果有的话需要修复
[x]补全模块测试，将项目中的所有的代码的模块化进行到底，确保代码的最高可复用程度
[x]installer脚本中添加一个如果没有python就下载python的逻辑，如果有python就升级python的逻辑，如果有python但是版本不符合就升级python的逻辑，确保安装的python版本是3.10+

[ ]前端的指标因为过滤的规则过于严格而无法显示的bug
[ ]高级参数设置的界面中有一些参数和既定的流程冲突或者冗余
[ ]后端数据链路的 4 个关键缺陷
[ ]1.  results.csv  扫描路径硬编码
文件： web/services/backend_metrics.py:44 
python
runs_dir = BASE_DIR / 'runs'

后端只扫描  runs/  目录，但你的训练配置（ output.project ）可能设置为  runs/detect 、 my_experiments  等其他路径。此时训练产出的  results.csv  不在扫描范围内，前端就永远找不到数据源。
[ ]2. 蒸馏指标列与 Ultralytics 存在"覆写竞争"
文件： core/distillation/adaptive_kd_trainer.py:778-827 
蒸馏指标（alpha、temperature、kd_loss）是通过回调  _on_fit_epoch_end  注入到  results.csv  的。但这里存在两个问题：
 
Ultralytics 的  final_eval()  会在训练结束时覆写整个  results.csv ，之前注入的蒸馏列可能被清空。
 
 _append_csv  使用  'w'  模式重写整个 CSV 文件，如果在训练过程中有并发读写，文件可能损坏。
虽然代码在  _on_train_end （第 959 行）设置了"安全网"来重新补全蒸馏数据，但如果训练被用户手动停止或异常中断，这个安全网不会执行。

[ ]3.  distill_log.json  回退机制失效
文件： web/services/backend_common.py:560-573 
python
has_distill_columns = any(col.startswith('distill/') for col in (columns or []))
distill_log_fallback = _load_distill_log_json(run_dir) if not has_distill_columns else []

这里的逻辑有缺陷：当 CSV 中有空列名但值为空时（例如 Ultralytics 覆写保留了列名但清空了值）， has_distill_columns  为  True ，导致系统不会加载  distill_log.json  作为备用数据。结果蒸馏图表没有任何数据点。
[ ]4. CSV 列名严格匹配，版本兼容性差
文件： web/services/backend_common.py:584-599 
python
chart['map_series']['map50'].append(_as_float(row.get('metrics/mAP50(B)')) or 0)
chart['lr_series']['pg0'].append(_as_float(row.get('lr/pg0')) or 0)

这些列名（如  metrics/mAP50(B) 、 lr/pg0 ）是硬编码的。如果 Ultralytics 版本升级导致列名变化（例如去掉括号、改变大小写），所有指标都会读取为  0  而不是真实值。此时图表会显示为全零直线，看起来就像"没数据"。

[ ]TraingingPanel拆解模块