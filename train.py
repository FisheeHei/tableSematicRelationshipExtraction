import argparse
import json
import logging
import os
import random
import shutil
import zipfile
from datetime import datetime

import numpy as np
import paddle
import paddle.nn as nn
import pandas as pd
from paddle.io import BatchSampler, DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from paddlenlp.transformers import AutoModel, AutoTokenizer, LinearDecayWithWarmup


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "Train_Set")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "cpa_output")
DEFAULT_SHORTCUT_NAME = "bert-base-cased"
TAIL_SAMPLER_HEAD_RATIO = 0.5
TAIL_SAMPLER_MAX_BOOST = 3.0
USE_FULL_FINAL_TRAINING = True


def ensure_train_directory(dir_path):
    if os.path.isdir(dir_path):
        return dir_path

    zip_path = f"{dir_path}.zip"
    if os.path.isfile(zip_path):
        extract_parent = os.path.dirname(dir_path)
        os.makedirs(extract_parent, exist_ok=True)
        logging.info(f"extract train dataset from {zip_path} to {extract_parent}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_parent)

    if os.path.isdir(dir_path):
        return dir_path

    raise ValueError(f"can't find: {dir_path}")


def load_data_from_directory(dir_path):
    dir_path = ensure_train_directory(dir_path)
    all_data = []
    csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]
    logging.info(f"load data from {dir_path} ...")

    for filename in tqdm(csv_files, desc=f"loading {os.path.basename(dir_path)}"):
        file_path = os.path.join(dir_path, filename)
        label_name = filename[:-4]
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding="utf-8-sig")
            if df.empty:
                continue
            df.columns = [str(col).strip() for col in df.columns]
            if "Subject" in df.columns and "Object" in df.columns:
                df = df[["Subject", "Object"]].dropna()
                df["label"] = label_name
                all_data.append(df)
        except Exception as e:
            logging.warning(f"{filename} load error: {e}")

    if not all_data:
        raise ValueError(f"{dir_path} not valid data")

    full_df = pd.concat(all_data, ignore_index=True)
    full_df["Subject"] = full_df["Subject"].astype(str)
    full_df["Object"] = full_df["Object"].astype(str)
    return full_df


def encode_pair(tokenizer, subject_text, object_text, max_length):
    try:
        encoding = tokenizer(
            text=subject_text,
            text_pair=object_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
    except TypeError:
        try:
            encoding = tokenizer(
                subject_text,
                object_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )
        except TypeError:
            encoding = tokenizer(
                text=subject_text,
                text_pair=object_text,
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


class RelationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_encoder, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.input_ids = []
        self.attention_masks = []
        self.token_type_ids = []
        self.label_ids = []

        iterator = tqdm(
            self.data.itertuples(index=False),
            total=len(self.data),
            desc="encoding data",
            leave=False,
        )
        for row in iterator:
            input_ids, attention_mask, token_type_ids = encode_pair(
                tokenizer,
                str(row.Subject),
                str(row.Object),
                max_length,
            )
            self.input_ids.append(input_ids)
            self.attention_masks.append(attention_mask)
            self.token_type_ids.append(token_type_ids)
            self.label_ids.append(np.int64(label_encoder.transform([row.label])[0]))

        self.input_ids = np.stack(self.input_ids).astype("int64")
        self.attention_masks = np.stack(self.attention_masks).astype("int64")
        self.token_type_ids = np.stack(self.token_type_ids).astype("int64")
        self.label_ids = np.array(self.label_ids, dtype="int64")

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, idx):
        return {
            "valid": True,
            "token_ids": self.input_ids[idx],
            "cls_mask": self.attention_masks[idx],
            "token_type_ids": self.token_type_ids[idx],
            "label_id": self.label_ids[idx],
        }


def dynamic_collate_fn(samples):
    valid_samples = [s for s in samples if s.get("valid", False)]
    if not valid_samples:
        return None

    return {
        "data": np.stack([s["token_ids"] for s in valid_samples]).astype("int64"),
        "label": np.array([s["label_id"] for s in valid_samples], dtype="int64"),
        "cls_mask": np.stack([s["cls_mask"] for s in valid_samples]).astype("int64"),
        "token_type_ids": np.stack([s["token_type_ids"] for s in valid_samples]).astype("int64"),
    }


class CPAModel(nn.Layer):
    def __init__(self, model_name, num_labels, use_flash_attn=False):
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
            raise ValueError("hidden_size in None")

        self.classifier = nn.Linear(hidden_size, num_labels)

        if use_flash_attn:
            logging.warning("flash attention not activate")

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(save_dir, "train.log"), mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def resolve_device(device_arg):
    try:
        custom_types = paddle.device.get_all_custom_device_type()
    except Exception:
        custom_types = []

    logging.info(f"custom device types: {custom_types}")

    if device_arg:
        try:
            dev = paddle.set_device(device_arg)
            logging.info(f"use: {dev}")
            return dev
        except Exception as e:
            logging.warning(f"{device_arg} use error: {e}")

    dev = paddle.set_device("cpu")
    logging.warning("set device to CPU")
    return dev


def save_label_classes(label_encoder, save_dir):
    path = os.path.join(save_dir, "label_classes.txt")
    with open(path, "w", encoding="utf-8") as f:
        for label in label_encoder.classes_:
            f.write(f"{label}\n")


def save_training_config(args, save_dir):
    config = {
        "shortcut_name": args.shortcut_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_ratio": args.val_ratio,
        "class_weight_power": args.class_weight_power,
        "max_grad_norm": args.max_grad_norm,
        "tail_sampler_head_ratio": TAIL_SAMPLER_HEAD_RATIO,
        "tail_sampler_max_boost": TAIL_SAMPLER_MAX_BOOST,
        "use_full_final_training": USE_FULL_FINAL_TRAINING,
    }
    with open(os.path.join(save_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def compute_competition_score(y_true, y_pred, num_classes):
    counts = np.bincount(y_true, minlength=num_classes).astype("float64")
    corrects = np.zeros(num_classes, dtype="float64")
    for true_id, pred_id in zip(y_true, y_pred):
        if true_id == pred_id:
            corrects[true_id] += 1.0

    valid_mask = counts > 0
    if not np.any(valid_mask):
        return 0.0

    valid_counts = counts[valid_mask]
    counts_max = float(valid_counts.max())
    counts_min = float(valid_counts.min())
    offset = counts_min * 0.1
    weights = (counts_max - counts + offset) / (counts_max + offset)
    scores = np.zeros(num_classes, dtype="float64")
    scores[valid_mask] = corrects[valid_mask] / counts[valid_mask]

    weighted_sum = float((weights[valid_mask] * scores[valid_mask]).sum())
    total_weight = float(weights[valid_mask].sum())
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def build_class_weights(label_encoder, label_counts, power):
    if power <= 0:
        return None

    counts = np.array([label_counts[label] for label in label_encoder.classes_], dtype="float32")
    counts_max = float(counts.max())
    counts_min = float(counts.min())
    offset = counts_min * 0.1
    weights = (counts_max - counts + offset) / (counts_max + offset)
    weights = np.power(weights, power).astype("float32")
    weights = weights / max(float(weights.mean()), 1e-8)
    return paddle.to_tensor(weights, dtype="float32")


def build_tail_aware_batch_sampler(train_df, batch_size):
    label_counts = train_df["label"].value_counts()
    class_sample_weights = np.sqrt(float(label_counts.max()) / label_counts)
    class_sample_weights = np.minimum(class_sample_weights, TAIL_SAMPLER_MAX_BOOST)

    # Mix boosted tail weights with the original distribution to avoid overfitting rare labels.
    class_sample_weights = TAIL_SAMPLER_HEAD_RATIO + (1.0 - TAIL_SAMPLER_HEAD_RATIO) * class_sample_weights
    sample_weights = train_df["label"].map(class_sample_weights).to_numpy(dtype="float64")

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df),
        replacement=True,
    )
    return BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)


def run_training(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, f"cpa_{timestamp}")
    setup_logging(save_dir)
    set_seed(args.random_seed)
    device = resolve_device(args.device)

    logging.info(f"device: {device}")

    raw_train_df = load_data_from_directory(args.train_dir)

    label_encoder = LabelEncoder()
    label_encoder.fit(raw_train_df["label"].unique())
    num_classes = len(label_encoder.classes_)
    logging.info(f"label_num: {num_classes}")
    save_label_classes(label_encoder, save_dir)
    save_training_config(args, save_dir)

    counts = raw_train_df["label"].value_counts()
    rare_labels = counts[counts < 2].index
    df_rare = raw_train_df[raw_train_df["label"].isin(rare_labels)]
    df_common = raw_train_df[~raw_train_df["label"].isin(rare_labels)]

    if len(df_common) == 0:
        raise ValueError("data num < 2, can't split dataset")

    train_c, val_c = train_test_split(
        df_common,
        test_size=args.val_ratio,
        stratify=df_common["label"],
        random_state=args.random_seed,
    )
    train_df = pd.concat([train_c, df_rare]).sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    val_df = val_c.reset_index(drop=True)
    logging.info(f"split success: train={len(train_df)}, val={len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    train_dataset = RelationDataset(train_df, tokenizer, label_encoder, args.max_length)
    train_batch_sampler = build_tail_aware_batch_sampler(train_df, args.batch_size)
    logging.info(
        f"use tail-aware sampler with max boost={TAIL_SAMPLER_MAX_BOOST:.1f} "
        f"and head ratio={TAIL_SAMPLER_HEAD_RATIO:.1f}"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=dynamic_collate_fn,
        num_workers=args.num_workers,
        return_list=True,
    )
    val_loader = DataLoader(
        RelationDataset(val_df, tokenizer, label_encoder, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dynamic_collate_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    model = CPAModel(args.shortcut_name, num_classes, args.use_flash_attention)
    total_steps = max(1, len(train_loader) * args.epoch)
    lr_scheduler = LinearDecayWithWarmup(args.lr, total_steps, warmup=args.warmup_ratio)
    grad_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm) if args.max_grad_norm > 0 else None
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        grad_clip=grad_clip,
    )
    class_weights = build_class_weights(label_encoder, counts, args.class_weight_power)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    if class_weights is not None:
        logging.info(f"use class weights with power={args.class_weight_power}")

    use_amp = args.use_amp and str(device) != "cpu"
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024) if use_amp else None

    best_score = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = args.patience

    logging.info("start training...")
    for epoch in range(args.epoch):
        model.train()
        tr_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")
        for batch in pbar:
            if batch is None:
                continue

            input_ids = paddle.to_tensor(batch["data"], dtype="int64")
            mask = paddle.to_tensor(batch["cls_mask"], dtype="int64")
            token_type_ids = paddle.to_tensor(batch["token_type_ids"], dtype="int64")
            label_ids = paddle.to_tensor(batch["label"], dtype="int64")

            if use_amp:
                with paddle.amp.auto_cast(enable=True):
                    logits = model(input_ids, mask, token_type_ids)
                    loss = loss_fn(logits, label_ids)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()
            else:
                logits = model(input_ids, mask, token_type_ids)
                loss = loss_fn(logits, label_ids)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

            lr_scheduler.step()
            loss_value = float(loss.numpy())
            tr_loss += loss_value
            train_steps += 1
            pbar.set_postfix({"loss": f"{loss_value:.4f}"})

        model.eval()
        val_preds = []
        val_labels = []
        with paddle.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                input_ids = paddle.to_tensor(batch["data"], dtype="int64")
                mask = paddle.to_tensor(batch["cls_mask"], dtype="int64")
                token_type_ids = paddle.to_tensor(batch["token_type_ids"], dtype="int64")
                label_ids = paddle.to_tensor(batch["label"], dtype="int64")

                if use_amp:
                    with paddle.amp.auto_cast(enable=True):
                        logits = model(input_ids, mask, token_type_ids)
                else:
                    logits = model(input_ids, mask, token_type_ids)

                preds = paddle.argmax(logits, axis=1)
                val_preds.extend(preds.numpy().tolist())
                val_labels.extend(label_ids.numpy().tolist())

        avg_train_loss = tr_loss / max(1, train_steps)
        val_score = compute_competition_score(val_labels, val_preds, num_classes)
        logging.info(f"Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | Val Score: {val_score:.4f}")

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            patience_counter = 0
            paddle.save(model.state_dict(), os.path.join(save_dir, "best_model.pdparams"))
            try:
                tokenizer.save_pretrained(save_dir)
            except Exception:
                pass
            logging.info(f"best model! (Score: {best_score:.4f})")
        else:
            patience_counter += 1
            logging.info(f"early stop count: {patience_counter}/{patience_limit}")
            if patience_counter == patience_limit:
                logging.info(f"{patience_limit} epoch not up, early stop!!!")
                break

    if USE_FULL_FINAL_TRAINING and best_epoch > 0:
        logging.info(f"start full-data final training for {best_epoch} epoch(s)...")
        set_seed(args.random_seed)
        best_model_path = os.path.join(save_dir, "best_model.pdparams")
        val_best_model_path = os.path.join(save_dir, "val_best_model.pdparams")
        if os.path.isfile(best_model_path):
            shutil.copyfile(best_model_path, val_best_model_path)
            logging.info(f"validation best model backed up to {val_best_model_path}")

        full_dataset = RelationDataset(raw_train_df, tokenizer, label_encoder, args.max_length)
        full_batch_sampler = build_tail_aware_batch_sampler(raw_train_df, args.batch_size)
        full_loader = DataLoader(
            full_dataset,
            batch_sampler=full_batch_sampler,
            collate_fn=dynamic_collate_fn,
            num_workers=args.num_workers,
            return_list=True,
        )

        final_model = CPAModel(args.shortcut_name, num_classes, args.use_flash_attention)
        final_total_steps = max(1, len(full_loader) * best_epoch)
        final_lr_scheduler = LinearDecayWithWarmup(args.lr, final_total_steps, warmup=args.warmup_ratio)
        final_grad_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm) if args.max_grad_norm > 0 else None
        final_optimizer = paddle.optimizer.AdamW(
            learning_rate=final_lr_scheduler,
            parameters=final_model.parameters(),
            grad_clip=final_grad_clip,
        )
        final_use_amp = args.use_amp and str(device) != "cpu"
        final_scaler = paddle.amp.GradScaler(init_loss_scaling=1024) if final_use_amp else None

        for final_epoch in range(best_epoch):
            final_model.train()
            final_loss = 0.0
            final_steps = 0
            pbar = tqdm(full_loader, desc=f"Full Train {final_epoch + 1}/{best_epoch}")
            for batch in pbar:
                if batch is None:
                    continue

                input_ids = paddle.to_tensor(batch["data"], dtype="int64")
                mask = paddle.to_tensor(batch["cls_mask"], dtype="int64")
                token_type_ids = paddle.to_tensor(batch["token_type_ids"], dtype="int64")
                label_ids = paddle.to_tensor(batch["label"], dtype="int64")

                if final_use_amp:
                    with paddle.amp.auto_cast(enable=True):
                        logits = final_model(input_ids, mask, token_type_ids)
                        loss = loss_fn(logits, label_ids)
                    scaled = final_scaler.scale(loss)
                    scaled.backward()
                    final_scaler.minimize(final_optimizer, scaled)
                    final_optimizer.clear_grad()
                else:
                    logits = final_model(input_ids, mask, token_type_ids)
                    loss = loss_fn(logits, label_ids)
                    loss.backward()
                    final_optimizer.step()
                    final_optimizer.clear_grad()

                final_lr_scheduler.step()
                loss_value = float(loss.numpy())
                final_loss += loss_value
                final_steps += 1
                pbar.set_postfix({"loss": f"{loss_value:.4f}"})

            logging.info(
                f"Full Train Epoch {final_epoch + 1}/{best_epoch} | "
                f"Loss: {final_loss / max(1, final_steps):.4f}"
            )

        paddle.save(final_model.state_dict(), best_model_path)
        logging.info("full-data final model saved as best_model.pdparams")

    logging.info(f"train finish, best score: {best_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shortcut_name", type=str, default=DEFAULT_SHORTCUT_NAME)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--class_weight_power", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    args = parser.parse_args()
    run_training(args)
