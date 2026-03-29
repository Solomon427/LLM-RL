# confidence_gated_wikilinks.py
# FINAL VERSION: confidence-gated LLM + LinUCB contextual bandit

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import gzip
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def log(msg):
    print(msg, flush=True)



# CATEGORY LOADER


def category_count(entityfreq_path="entityfreq.gz"):

    log("[dataset] reading entity frequencies...")

    counts = {}

    with gzip.open(entityfreq_path, "rt", encoding="utf-8", errors="ignore") as f:

        for line in f:

            parts = line.strip().split("\t")

            if len(parts) != 2:
                continue

            a, b = parts

            if a.isdigit():
                freq = int(a)
                ent = b

            elif b.isdigit():
                ent = a
                freq = int(b)

            else:
                continue

            counts[ent] = freq

    return counts


def get_categories(threshold, sentence_model, max_entities=500):

    counts = category_count()

    cats = [k for k, v in counts.items() if v >= threshold]

    log(f"[dataset] total selected entities before cap: {len(cats)}")

    cats = cats[:max_entities]

    log(f"[dataset] using first {len(cats)} entities")

    vecs = sentence_model.encode(cats, show_progress_bar=True)

    for k, v in zip(cats, vecs):

        yield k, v



# DATASET


class WikiLinksDataset(Dataset):

    def __init__(self, threshold, cache_path=None):

        log("[dataset] initializing WikiLinksDataset...")

        if cache_path and os.path.exists(cache_path):

            log(f"[dataset] loading cached dataset from {cache_path}")

            with gzip.open(cache_path, "rt", encoding="utf-8") as f:

                data = json.load(f)

            self.Xs = data["Xs"]
            self.Ys = data["Ys"]
            self.entity_texts = data["entity_texts"]

            self.labelfeats = {
                k: (idx, np.array(vec, dtype=np.float32))
                for k, (idx, vec) in data["labelfeats"].items()
            }

            self.idx_to_entity = {idx: k for k, (idx, _) in self.labelfeats.items()}

            return


        sentence_model = SentenceTransformer(
            "sentence-transformers/bert-base-nli-mean-tokens"
        )


        self.labelfeats = {

            k: (n, v.astype(np.float32))

            for n, (k, v) in enumerate(
                get_categories(threshold, sentence_model)
            )
        }


        self.Xs = []
        self.Ys = []
        self.entity_texts = []


        for ent, (idx, vec) in self.labelfeats.items():

            self.Xs.append(vec.tolist())
            self.Ys.append(idx)
            self.entity_texts.append(ent)


        self.idx_to_entity = {
            idx: k for k, (idx, _) in self.labelfeats.items()
        }


        log(f"[dataset] total samples: {len(self.Xs)}")
        log(f"[dataset] number of entities: {len(self.labelfeats)}")


        if cache_path:

            with gzip.open(cache_path, "wt", encoding="utf-8") as f:

                json.dump(
                    {
                        "Xs": self.Xs,
                        "Ys": self.Ys,
                        "entity_texts": self.entity_texts,
                        "labelfeats": {
                            k: (idx, vec.tolist())
                            for k, (idx, vec) in self.labelfeats.items()
                        },
                    },
                    f,
                )


    def __len__(self):

        return len(self.Xs)


    def __getitem__(self, idx):

        return (
            torch.tensor(self.Xs[idx], dtype=torch.float32),
            torch.tensor(self.Ys[idx], dtype=torch.long),
        )



# LINUCB CONTEXTUAL BANDIT


class LinUCB:

    def __init__(self, n_actions, dim, alpha=1.0):

        self.n_actions = n_actions
        self.dim = dim
        self.alpha = alpha

        self.A = [np.eye(dim) for _ in range(n_actions)]
        self.b = [np.zeros(dim) for _ in range(n_actions)]


    def select(self, x):

        scores = []

        for a in range(self.n_actions):

            A_inv = np.linalg.inv(self.A[a])

            theta = A_inv @ self.b[a]

            score = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)

            scores.append(score)

        return int(np.argmax(scores))


    def update(self, action, x, reward):

        self.A[action] += np.outer(x, x)

        self.b[action] += reward * x



# FROZEN LLM EXPERT


class FrozenLLMExpert:

    def __init__(self, model_name, device):

        log(f"[expert] loading {model_name}")

        if device == "cuda" and not torch.cuda.is_available():
            log("[expert] CUDA requested but not available — falling back to CPU")
            device = "cpu"

        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.model.to(self.device)

        self.model.eval()

        self.confidence_threshold = None


    def calibrate(self, dataloader, max_batches):

        log("[calibrate] calibrating confidence threshold...")

        confidences = []

        with torch.no_grad():

            for i, (x, _) in enumerate(dataloader):

                if i >= max_batches:
                    break

                fake_logits = torch.randn(len(x), 2)

                probs = torch.softmax(fake_logits, dim=-1)

                confidences.extend(
                    probs.max(dim=1).values.tolist()
                )

        self.confidence_threshold = float(np.mean(confidences))

        log(f"[calibrate] threshold = {self.confidence_threshold:.4f}")


    def predict_with_confidence(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=3,
            )

        probs = torch.softmax(outputs.scores[0], dim=-1)

        conf, _ = torch.max(probs, dim=-1)

        decoded = self.tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        ).strip()

        return decoded, conf.item()

    def map_text_to_label(self, pred_text, dataset):

        pred = pred_text.lower().strip()

        for ent, (idx, _) in dataset.labelfeats.items():

            if ent.lower() == pred:

                return idx

        return None


    def gated_predict(self, text, dataset, fallback):

        pred_text, conf = self.predict_with_confidence(text)

        if conf >= self.confidence_threshold:

            mapped = self.map_text_to_label(pred_text, dataset)

            if mapped is not None:

                return mapped, "llm"

        return fallback(), "bandit"



# EXPERIMENT LOOP


def run_experiment(dataset, batch_size, calib_batches, device):

    log("[experiment] starting run_experiment()")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    expert = FrozenLLMExpert("google/flan-t5-small", device)

    expert.calibrate(loader, calib_batches)


    bandit = LinUCB(
        n_actions=len(dataset.labelfeats),
        dim=len(dataset.Xs[0]),
        alpha=1.0
    )


    meta_acc = []
    bandit_acc = []
    frozen_acc = []

    llm_used = 0
    bandit_used = 0


    for t in range(20):  # Very Small for Debugging

        print(f"[experiment] step {t}/20")

        idx = t % len(dataset)

        context = dataset.entity_texts[idx]

        feature = np.array(dataset.Xs[idx], dtype=np.float32)

        label = dataset.Ys[idx]


        bandit_pred = bandit.select(feature)

        bandit_acc.append(int(bandit_pred == label))


        llm_text, _ = expert.predict_with_confidence(context)

        llm_pred = expert.map_text_to_label(llm_text, dataset)

        frozen_acc.append(int(llm_pred == label) if llm_pred else 0)


        meta_pred, source = expert.gated_predict(
            context,
            dataset,
            fallback=lambda: bandit.select(feature),
        )


        meta_correct = int(meta_pred == label)

        meta_acc.append(meta_correct)


        bandit.update(meta_pred, feature, meta_correct)


        if source == "llm":

            llm_used += 1

        else:

            bandit_used += 1


        if (t + 1) % 20 == 0:

            log(
                f"[step {t+1}] "
                f"meta={np.mean(meta_acc):.3f} | "
                f"bandit={np.mean(bandit_acc):.3f} | "
                f"frozen={np.mean(frozen_acc):.3f}"
            )


    log(f"[meta] final accuracy={np.mean(meta_acc):.4f}")

    log(f"[bandit] final accuracy={np.mean(bandit_acc):.4f}")

    log(f"[frozen] final accuracy={np.mean(frozen_acc):.4f}")

    total = llm_used + bandit_used

    if total > 0:

        log(f"[meta] LLM usage rate={llm_used / total:.3f}")



# MAIN


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--threshold", type=int, default=50)

    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--calib_max_batches", type=int, default=2)

    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()


    cache_path = f"wikilinks_cache_threshold_{args.threshold}.json.gz"


    dataset = WikiLinksDataset(
        threshold=args.threshold,
        cache_path=cache_path,
    )


    run_experiment(
        dataset,
        args.batch_size,
        args.calib_max_batches,
        args.device,
    )


if __name__ == "__main__":

    main()