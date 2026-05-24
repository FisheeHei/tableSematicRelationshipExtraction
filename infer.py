import argparse
import json
import os

import numpy as np
import paddle
import paddle.nn as nn
import pandas as pd
from paddle.io import DataLoader, Dataset
from tqdm import tqdm

from paddlenlp.transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_CSV = os.path.join(PROJECT_ROOT, "dataset", "test.csv")
DEFAULT_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "submission.csv")
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "cpa_output")
DEFAULT_SHORTCUT_NAME = "bert-base-cased"
DEFAULT_MAX_LENGTH = 128


def find_latest_artifact(filename):
    if not os.path.isdir(DEFAULT_OUTPUT_ROOT):
        return None

    candidates = []
    for entry in os.listdir(DEFAULT_OUTPUT_ROOT):
        artifact_path = os.path.join(DEFAULT_OUTPUT_ROOT, entry, filename)
        if os.path.isfile(artifact_path):
            candidates.append(artifact_path)

    if not candidates:
        return None

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


DEFAULT_MODEL_PATH = find_latest_artifact("best_model.pdparams") or os.path.join(
    DEFAULT_OUTPUT_ROOT, "cpa_20260422_191904", "best_model.pdparams"
)
DEFAULT_LABELS_PATH = find_latest_artifact("label_classes.txt") or os.path.join(
    PROJECT_ROOT, "dataset", "labels.txt"
)


class CPAModel(nn.Layer):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        hidden_size = None
        if hasattr(self.encoder, "config"):
            if isinstance(self.encoder.config, dict):
                hidden_size = self.encoder.config.get("hidden_size", None)
            else:
                hidden_size = getattr(self.encoder.config, "hidden_size", None)

        if hidden_size is None and hasattr(self.encoder, "embeddings") and hasattr(
            self.encoder.embeddings, "word_embeddings"
        ):
            hidden_size = self.encoder.embeddings.word_embeddings.weight.shape[-1]

        if hidden_size is None:
            raise ValueError("Unable to infer hidden_size automatically. Please check the pretrained model.")

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**model_inputs)

        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs

        cls_embedding = sequence_output[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding))
        return logits


def encode_pair(tokenizer, text_a, text_b, max_length):
    try:
        encoding = tokenizer(
            text=text_a,
            text_pair=text_b,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
    except TypeError:
        try:
            encoding = tokenizer(
                text_a,
                text_b,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )
        except TypeError:
            encoding = tokenizer(
                text=text_a,
                text_pair=text_b,
                max_seq_len=max_length,
                pad_to_max_seq_len=True,
                truncation=True,
                return_attention_mask=True,
            )

    input_ids = encoding["input_ids"]
    attention_mask = encoding.get("attention_mask", None)
    token_type_ids = encoding.get("token_type_ids", None)

    if attention_mask is None:
        seq_len = encoding.get("seq_len", len(input_ids))
        seq_len = min(seq_len, max_length)
        attention_mask = [1] * seq_len + [0] * (max_length - seq_len)

    if token_type_ids is None:
        token_type_ids = [0] * len(input_ids)

    return (
        np.array(input_ids, dtype="int64"),
        np.array(attention_mask, dtype="int64"),
        np.array(token_type_ids, dtype="int64"),
    )


class SingleTableInferenceDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.original_rows = []

        df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8-sig")
        df.columns = [str(col).strip() for col in df.columns]

        subject_col = None
        object_col = None
        for col in df.columns:
            if col.lower() == "subject":
                subject_col = col
            elif col.lower() == "object":
                object_col = col

        if subject_col is None or object_col is None:
            raise ValueError("The CSV file must contain 'Subject' and 'Object' columns (case-insensitive).")

        temp_df = df[[subject_col, object_col]].dropna()
        for idx, row in temp_df.iterrows():
            self.samples.append((str(row[subject_col]), str(row[object_col])))
            self.original_rows.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subject_text, object_text = self.samples[idx]
        input_ids, attention_mask, token_type_ids = encode_pair(
            self.tokenizer,
            subject_text,
            object_text,
            self.max_length,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "orig_idx": np.int64(idx),
        }


def collate_fn(samples):
    return {
        "input_ids": np.stack([s["input_ids"] for s in samples]).astype("int64"),
        "attention_mask": np.stack([s["attention_mask"] for s in samples]).astype("int64"),
        "token_type_ids": np.stack([s["token_type_ids"] for s in samples]).astype("int64"),
        "orig_idx": np.array([s["orig_idx"] for s in samples], dtype="int64"),
    }


def resolve_device(device_arg):
    try:
        custom_types = paddle.device.get_all_custom_device_type()
    except Exception:
        custom_types = []

    print(f"Available custom devices: {custom_types}")

    if device_arg:
        try:
            dev = paddle.set_device(device_arg)
            print(f"Using requested device: {dev}")
            return dev
        except Exception as e:
            print(f"Failed to set requested device {device_arg}: {e}")

    dev = paddle.set_device("cpu")
    print("Falling back to CPU.")
    return dev


def load_training_config(model_path):
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "training_config.json")
    if not os.path.isfile(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_inference(args):
    device = resolve_device(args.device)
    training_config = load_training_config(args.model_path)
    shortcut_name = args.shortcut_name or training_config.get("shortcut_name", DEFAULT_SHORTCUT_NAME)
    max_length = args.max_length or training_config.get("max_length", DEFAULT_MAX_LENGTH)

    with open(args.labels_path, "r", encoding="utf-8-sig") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    id2label = {idx: label for idx, label in enumerate(classes)}

    tokenizer = AutoTokenizer.from_pretrained(shortcut_name)
    model = CPAModel(shortcut_name, len(classes))

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    state_dict = paddle.load(args.model_path)
    model.set_state_dict(state_dict)
    model.eval()

    dataset = SingleTableInferenceDataset(args.input_csv, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        return_list=True,
    )

    print(f"Starting inference. Total valid rows: {len(dataset)}")
    predictions = [None] * len(dataset)
    use_amp = args.use_amp and str(device) != "cpu"

    with paddle.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            ids = paddle.to_tensor(batch["input_ids"], dtype="int64")
            mask = paddle.to_tensor(batch["attention_mask"], dtype="int64")
            token_type_ids = paddle.to_tensor(batch["token_type_ids"], dtype="int64")
            orig_indices = batch["orig_idx"].tolist()

            if use_amp:
                with paddle.amp.auto_cast(enable=True):
                    logits = model(ids, mask, token_type_ids)
            else:
                logits = model(ids, mask, token_type_ids)

            preds = paddle.argmax(logits, axis=1).numpy().tolist()
            for idx_in_batch, pred_idx in enumerate(preds):
                original_position = orig_indices[idx_in_batch]
                predictions[original_position] = id2label[pred_idx]

    original_df = pd.read_csv(args.input_csv, low_memory=False, encoding="utf-8-sig")
    original_df.columns = [str(col).strip() for col in original_df.columns]

    subject_col = None
    object_col = None
    for col in original_df.columns:
        if col.lower() == "subject":
            subject_col = col
        elif col.lower() == "object":
            object_col = col

    if subject_col is None or object_col is None:
        raise ValueError("The CSV file must contain 'Subject' and 'Object' columns (case-insensitive).")

    valid_mask = original_df[subject_col].notna() & original_df[object_col].notna()
    valid_indices = original_df[valid_mask].index.tolist()

    original_df["Label"] = None
    for row_idx, pred_label in zip(valid_indices, predictions):
        original_df.loc[row_idx, "Label"] = pred_label

    original_df.to_csv(args.output_file, index=False, encoding="utf-8-sig")
    print(f"Inference completed. Results saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--labels_path", type=str, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--shortcut_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--use_amp", action="store_true")
    args = parser.parse_args()
    run_inference(args)
