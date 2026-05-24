# Table Semantic Relationship Extraction

本仓库是表格语义关系抽取任务的 Paddle/PaddleNLP 版本，基于原始 Baseline 进行了路径整理、训练流程优化和 AIStudio Notebook 运行适配。

> 原始仓库：[nanhuiNana/tableSematicRelationshipExtraction](https://github.com/nanhuiNana/tableSematicRelationshipExtraction)
>
> 当前 Fork：[FisheeHei/tableSematicRelationshipExtraction](https://github.com/FisheeHei/tableSematicRelationshipExtraction)

## 运行方式

本项目在飞桨 AIStudio 上运行时，推荐且默认只通过 `main.ipynb` 启动。

`main.ipynb` 会依次执行：

1. 添加 `/home/aistudio/external-libraries` 到 Python 搜索路径。
2. 运行 `train.py` 完成训练。
3. 运行 `infer.py` 生成 `submission.csv`。

核心启动参数示例：

```python
subprocess.run([
    sys.executable, "train.py",
    "--shortcut_name", "bert-base-cased",
    "--max_grad_norm", "0.5",
    "--use_amp",
], check=True)

subprocess.run([
    sys.executable, "infer.py",
    "--device", "gpu",
    "--use_amp",
], check=True)
```

## 数据目录结构

数据文件不建议直接提交到 GitHub。请在运行前手动放置数据，并确保目录结构如下：

```text
/home/aistudio/
├── main.ipynb
├── train.py
├── infer.py
├── dataset/
│   ├── test.csv
│   ├── labels.txt
│   ├── Train_Set.zip
│   └── Train_Set/
│       ├── relation_1.csv
│       ├── relation_2.csv
│       └── ...
└── external-libraries/
```

特别注意：

- `dataset/Train_Set.zip` 需要解压成 `dataset/Train_Set/` 后再训练。
- `Train_Set/` 内部通常包含大量按关系名命名的 CSV 文件，解压后的内容可能较多，不建议上传到 GitHub。
- `train.py` 默认读取 `dataset/Train_Set`；如果只存在 `dataset/Train_Set.zip`，代码会尝试自动解压。
- `infer.py` 默认读取 `dataset/test.csv`，并输出 `submission.csv`。

## 主要文件

- `main.ipynb`：AIStudio 一键运行入口。
- `train.py`：训练脚本，包含比赛指标验证、长尾采样、类别权重、AMP、梯度裁剪和全量最终训练流程。
- `infer.py`：推理脚本，自动读取最新训练输出并生成提交文件。
- `environment/requirements.txt`：依赖版本参考。
- `baseline/`：原始 baseline 脚本参考。

## 训练机制说明

当前训练流程主要包括：

- 使用 `bert-base-cased` 作为默认训练模型。
- 使用 PaddleNLP `AutoTokenizer` / `AutoModel`。
- 使用 Subject/Object 成对编码，而不是简单文本拼接。
- 使用与比赛一致的少样本加权指标选择验证集最佳 epoch。
- 使用 tail-aware sampler 提高长尾关系类型在训练中的采样概率。
- 使用类别权重 CrossEntropyLoss。
- 支持 AMP 混合精度训练。
- 先通过验证集确定最佳 epoch，再用全量训练集进行最终训练并保存 `best_model.pdparams`。

## 输出文件

训练后默认生成：

```text
cpa_output/
└── cpa_YYYYMMDD_HHMMSS/
    ├── best_model.pdparams
    ├── val_best_model.pdparams
    ├── label_classes.txt
    ├── training_config.json
    └── train.log
```

推理后默认生成：

```text
submission.csv
```

这些运行产物默认不会提交到 GitHub。

## 本地依赖安装

如需在 Notebook 中安装依赖，可参考：

```python
!pip install -r environment/requirements.txt -t /home/aistudio/external-libraries
```

在 AIStudio 中如果环境已经包含 Paddle/PaddleNLP，可优先使用平台预装环境，避免重复安装导致版本冲突。

## 备注

本仓库重点是比赛运行代码和 AIStudio 运行流程。由于训练数据可能较大，仓库中只保留代码、Notebook 和依赖说明；数据请通过比赛页面或平台数据集挂载方式准备。
