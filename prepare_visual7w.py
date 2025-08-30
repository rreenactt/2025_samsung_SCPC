#!/usr/bin/env python
"""
prepare_visual7w.py  (v2, 2025-07)
────────────────────────────────────────────────────────
Visual7W telling 데이터 전자동 준비:

  • telling.json (≈55 MB)
  • dataset_v7w_telling.zip  (24 MB → 정답·선지)
  • visual7w_images.zip      (≈1.7 GB COCO 이미지만 추린 버전)¹
  • 압축 해제 & TSV 생성

미러 순서
  1) Stanford  : https://ai.stanford.edu/~yukez/…   (구·원본, 종종 404)
  2) MIT       : http://visual7w.csail.mit.edu/…
  3) HyperAI   : https://hyper.ai/…/Visual7W/data/…

CLI
  python prepare_visual7w.py --root D:/data/visual7w
"""
from __future__ import annotations
import os, sys, json, csv, zipfile, argparse, hashlib, requests, itertools
from pathlib import Path
from tqdm import tqdm

MIRRORS = {
    "json": [
        # (name, url)
        ("stanford", "https://ai.stanford.edu/~yukez/visual7w/telling.json"),
        ("mit"     , "http://visual7w.csail.mit.edu/telling.json"),
        ("hyperai" , "https://hyper.ai/api/file/Visual7W/data/dataset_v7w_telling.json"),
    ],
    "zip" : [
        ("stanford", "https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip"),
        ("mit"     , "http://visual7w.csail.mit.edu/dataset_v7w_telling.zip"),
        ("hyperai" , "https://hyper.ai/api/file/Visual7W/data/dataset_v7w_telling.zip"),
    ],
    "img" : [
        ("stanford", "https://ai.stanford.edu/~yukez/visual7w_images.zip"),
        ("hyperai" , "https://hyper.ai/api/file/Visual7W/data/visual7w_images.zip"),
    ],
}

CHUNK = 1 << 20  # 1 MiB


# ───────────────────────── helpers ──────────────────────────
def md5sum(path: Path) -> str:
    h = hashlib.md5(); buf = memoryview(bytearray(CHUNK))
    with path.open("rb") as f:
        for n in iter(lambda: f.readinto(buf), 0):
            h.update(buf[:n])
    return h.hexdigest()

def download_try(urls: list[tuple[str,str]], dest: Path):
    """순서대로 시도→성공 시 dest 저장, 이어받기 지원"""
    for name, url in urls:
        try:
            with requests.get(url, stream=True, timeout=15) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"{r.status_code}")
                total = int(r.headers.get("content-length", 0))
                mode  = "ab" if dest.exists() else "wb"
                done  = dest.stat().st_size if dest.exists() else 0
                if done == total and total > 0:
                    print(f"✓ 이미 완료  [{name}] {dest.name}")
                    return
                if done > 0:
                    headers = {"Range": f"bytes={done}-"}
                    r = requests.get(url, headers=headers, stream=True, timeout=15)
                with dest.open(mode) as f, tqdm(total=total, initial=done,
                                                 unit="B", unit_scale=True,
                                                 desc=f"⇣ {dest.name} ({name})") as bar:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk); bar.update(len(chunk))
            return
        except Exception as e:
            print(f"⚠  실패 [{name}] {url.split('//')[1].split('/')[0]}  ({e})")
    raise RuntimeError(f"✗ 모든 미러 실패 → {dest.name}")

# ───────────────────────── main ──────────────────────────
def main(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": root / "telling.json",
        "zip" : root / "dataset_v7w_telling.zip",
        "img" : root / "visual7w_images.zip",
        "tsv" : root / "visual7w_telling.tsv",
        "img_dir": root / "images",
    }

    # 1) JSON + ZIP + images.zip
    download_try(MIRRORS["json"], paths["json"])
    download_try(MIRRORS["zip"],  paths["zip"])
    if not paths["img_dir"].exists():          # 이미지 풀린 폴더가 없으면 ZIP 시도
        download_try(MIRRORS["img"], paths["img"])

    # 2) unzip (필요한 것만)
    if not paths["img_dir"].exists():
        print("⇡ 이미지 ZIP 해제 중 …")
        with zipfile.ZipFile(paths["img"]) as zf:
            zf.extractall(root)
    if not (root/"dataset_v7w_telling").exists():
        with zipfile.ZipFile(paths["zip"]) as zf:
            zf.extractall(root)

    # 3) TSV 생성
    if paths["tsv"].exists():
        print("✓ TSV 이미 존재:", paths['tsv']); return

    qa_json = root/"dataset_v7w_telling"/"dataset_v7w_telling.json"
    with qa_json.open(encoding="utf-8") as f:
        qa_data = json.load(f)

    with paths["tsv"].open("w", newline="", encoding="utf-8") as fout:
        wr = csv.writer(fout, delimiter="\t")
        wr.writerow(["img_path","question","A","B","C","D","label"])

        miss = 0
        for obj in qa_data["questions"]:
            img_path = paths["img_dir"] / obj["image_filename"]
            if not img_path.exists():
                miss += 1; continue
            choices = obj["multiple_choices"]
            wr.writerow([str(img_path), obj["question"], *choices,
                         chr(65 + obj["answer_idx"])])
    print(f"✅ TSV 완성  → {paths['tsv']}  (누락 이미지 {miss})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="visual7w_data", help="저장 폴더")
    main(Path(p.parse_args().root))
