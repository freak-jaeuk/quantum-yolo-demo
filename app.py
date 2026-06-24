"""
Quantum-YOLO Demo — edge object detection with the DEPLOYMENT artifacts.

Self-contained demo of the end product: a YOLO26n whose attention is augmented
by a genuine variational quantum circuit (unitary-verified + run on a real IBM
QPU), then COMPILED to classical ops and exported to ONNX / INT8 for the edge.

Runs fully standalone from this folder (no COCO dataset, no GPU required):
  pip install -r requirements.txt
  python app.py            # then open http://localhost:7861

Tabs:
  - 이미지   : run one deployment form on an image
  - 비교     : run all three forms (INT8 / fp32-ONNX / PyTorch-VQC) on ONE image
  - 영상     : run on an uploaded video (CCTV-style)
  - 실시간   : live webcam stream detection (real-time, runs in the browser)

GPU is optional. To enable it on a Linux box with the real driver libs:
  LD_LIBRARY_PATH=/usr/lib64 python app.py
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import gradio as gr

from yolo26q.register import register_quantum_modules
register_quantum_modules()
from models.vqc_deploy import replace_vqcs_with_compiled
from ultralytics import YOLO

# ---- deployment artifacts (bundled in ./weights) ----
# Full-COCO (118k imgs, 80 cls) YOLO26n + QuantumC2PSA, converged @ epoch 40.
# full-val mAP50-95: fp32 0.3635 / INT8 0.3594 (-1.13%). CPU FPS (Xeon): fp32 63 /
# INT8 21; GPU(.pt) 189. INT8 win here = SIZE (9.98->3.58MB), not CPU speed
# (ONNXRuntime QDQ overhead, no VNNI int8 kernels); speed win shows on OpenVINO/VNNI/NPU.
WDIR = str(ROOT / "weights")
CKPTS = {
    "INT8 (배포형, 3.6MB)":      f"{WDIR}/best_int8_pctA.onnx",
    "fp32 (컴파일 ONNX, 10MB)":  f"{WDIR}/best.onnx",
    "PyTorch (VQC)":             f"{WDIR}/best.pt",
}
SAMPLE_IMG_DIR = ROOT / "samples" / "images"
SAMPLE_VID_DIR = ROOT / "samples" / "video"
_models = {}


def _gpu_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


GPU_OK = _gpu_ok()


def get_model(name):
    if name not in _models:
        path = CKPTS[name]
        if path.endswith(".onnx"):
            m = YOLO(path, task="detect")            # ONNXRuntime (CPU) backend
        else:
            m = YOLO(path)
            replace_vqcs_with_compiled(m.model)      # VQC -> classical (edge form)
            m.model.float()
        _models[name] = m
    return _models[name]


def _resolve_device(model_name, device_choice):
    # ONNX models run on CPU (onnxruntime CPU build); only the .pt can use CUDA.
    if device_choice == "GPU" and GPU_OK and CKPTS[model_name].endswith(".pt"):
        return 0
    return "cpu"


def _size_mb(name):
    return os.path.getsize(CKPTS[name]) / 1e6


def detect_image(image, model_name, conf, device_choice):
    if image is None:
        return None, "이미지를 업로드하세요."
    m = get_model(model_name)
    dev = _resolve_device(model_name, device_choice)
    t0 = time.time()
    res = m.predict(image, conf=conf, device=dev, half=False, verbose=False)[0]
    dt = (time.time() - t0) * 1000
    annotated = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
    n = len(res.boxes) if res.boxes is not None else 0
    devlabel = "GPU" if dev == 0 else "CPU"
    info = (f"{model_name}  ({_size_mb(model_name):.1f} MB)\n"
            f"탐지 객체: {n}개\n추론: {dt:.0f} ms ({1000/dt:.1f} FPS, {devlabel})")
    return annotated, info


def detect_compare(image, conf):
    """Run all three deployment forms on ONE image (all on CPU = edge realism).
    Demonstrates the value prop: the 3.6 MB INT8 model ≈ fp32 detections."""
    if image is None:
        return None, None, None, "이미지를 업로드하세요."
    outs, rows = [], []
    for name in CKPTS:
        m = get_model(name)
        t0 = time.time()
        res = m.predict(image, conf=conf, device="cpu", half=False, verbose=False)[0]
        dt = (time.time() - t0) * 1000
        outs.append(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB))
        n = len(res.boxes) if res.boxes is not None else 0
        rows.append(f"| {name} | {_size_mb(name):.1f} | {n} | {dt:.0f} | {1000/dt:.1f} |")
    table = ("**같은 이미지, 세 배포 형태 (모두 CPU)** — INT8(3.6MB)이 fp32와 사실상 동일 탐지\n\n"
             "| 모델 | 크기(MB) | 탐지수 | 시간(ms) | FPS |\n"
             "|---|---|---|---|---|\n" + "\n".join(rows))
    return outs[0], outs[1], outs[2], table


def detect_video(video_path, model_name, conf, max_seconds, device_choice):
    if video_path is None:
        return None, "영상을 업로드하세요."
    m = get_model(model_name)
    dev = _resolve_device(model_name, device_choice)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * max_seconds)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = str(out_dir / "qyolo_out.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    n_frames, total_det, t0 = 0, 0, time.time()
    while n_frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        res = m.predict(frame, conf=conf, device=dev, half=False, verbose=False)[0]
        total_det += len(res.boxes) if res.boxes is not None else 0
        writer.write(res.plot())
        n_frames += 1
    cap.release()
    writer.release()

    # Re-encode mp4v -> H.264 (avc1) so browsers can play it inline.
    h264_path = str(out_dir / "qyolo_out_h264.mp4")
    try:
        import imageio_ffmpeg, subprocess
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff, "-y", "-i", out_path, "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        h264_path], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        play_path = h264_path
    except Exception:
        play_path = out_path

    dt = time.time() - t0
    proc_fps = n_frames / dt if dt > 0 else 0
    devlabel = "GPU" if dev == 0 else "CPU"
    info = (f"{model_name}\n처리 프레임: {n_frames}\n"
            f"평균 탐지: {total_det/max(n_frames,1):.1f} 객체/프레임\n"
            f"처리 속도: {proc_fps:.1f} FPS ({devlabel})")
    return play_path, info


def detect_stream(frame, model_name, conf):
    """Live webcam frame -> annotated frame + per-frame FPS. Runs on CPU (edge
    realism); the .pt form uses GPU if available. gradio sends RGB numpy frames."""
    if frame is None:
        return None, "웹캠을 시작하세요."
    m = get_model(model_name)
    dev = _resolve_device(model_name, "GPU")  # use GPU only if .pt + available
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    t0 = time.time()
    res = m.predict(bgr, conf=conf, device=dev, half=False, verbose=False)[0]
    dt = (time.time() - t0) * 1000
    out = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
    n = len(res.boxes) if res.boxes is not None else 0
    devlabel = "GPU" if dev == 0 else "CPU"
    return out, f"실시간 탐지 {n}개 · {dt:.0f} ms ({1000/dt:.1f} FPS, {devlabel})"


def _sample_images():
    return sorted(str(p) for p in SAMPLE_IMG_DIR.glob("*.jpg"))


def _sample_videos():
    return sorted(str(p) for p in SAMPLE_VID_DIR.glob("*.mp4")
                  if "detected" not in p.name)


DEV_CHOICES = ["CPU"] + (["GPU"] if GPU_OK else [])

with gr.Blocks(title="Quantum-YOLO Demo") as demo:
    gr.Markdown(
        "# 🔬 Quantum-YOLO 엣지 탐지 데모\n"
        "YOLO26 어텐션을 **변분 양자 회로(VQC)** 로 학습 → 회로를 **고전 연산으로 컴파일**(무손실) "
        "→ **ONNX / INT8** export → *실제 배포 아티팩트*를 구동합니다.\n\n"
        "**파이프라인:** 양자 학습 → 고전 컴파일(0 추론오버헤드) → INT8(3.6MB) → 엣지 배포  \n"
        "- 양자 회로는 유니터리 검증 + **실제 IBM 양자컴퓨터(ibm_fez)에서 실행 검증** (cosine 0.978)\n"
        "- 세 배포 형태는 **사실상 동일한 탐지**(컴파일=무손실, INT8≈fp32)이며 크기·속도만 다름\n"
        "- **실시간 가능**: GPU 189 FPS · CPU(fp32) 63 FPS · CPU(INT8) 21 FPS (Xeon 기준) — 모두 영상 프레임레이트 초과\n"
        "- 정직성: 이 CPU에선 INT8 이득은 **속도가 아닌 크기**(9.98→3.58MB). 속도 이득은 OpenVINO/VNNI/NPU에서 발현\n"
        f"- {'🟢 GPU 사용 가능' if GPU_OK else '⚪ CPU 전용 (노트북 CPU에서도 동작 — 속도는 사양에 따라 다름)'}\n"
        "- ✅ **풀COCO(118k, 80클래스) 학습 모델** — full-val mAP50-95 0.364 (fp32) / 0.359 (INT8)"
    )
    conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="confidence 임계값")

    with gr.Tab("이미지"):
        with gr.Row():
            model_name = gr.Dropdown(list(CKPTS.keys()),
                                     value="INT8 (배포형, 3.6MB)", label="모델 (배포 형태)")
            dev_img = gr.Dropdown(DEV_CHOICES, value="CPU", label="장치")
        with gr.Row():
            img_in = gr.Image(type="numpy", label="입력 이미지")
            img_out = gr.Image(label="탐지 결과")
        img_info = gr.Textbox(label="정보", lines=3)
        gr.Button("탐지 실행", variant="primary").click(
            detect_image, [img_in, model_name, conf, dev_img], [img_out, img_info])
        if _sample_images():
            gr.Examples(_sample_images(), img_in, label="샘플 이미지")

    with gr.Tab("비교 (INT8 vs fp32 vs VQC)"):
        gr.Markdown("같은 이미지에 **세 배포 형태**를 동시 실행 → INT8(3.6MB)이 fp32와 거의 동일 탐지임을 확인")
        cmp_in = gr.Image(type="numpy", label="입력 이미지")
        with gr.Row():
            cmp_int8 = gr.Image(label="INT8 (3.6MB)")
            cmp_fp32 = gr.Image(label="fp32 컴파일 ONNX")
            cmp_pt = gr.Image(label="PyTorch VQC")
        cmp_table = gr.Markdown()
        gr.Button("3개 비교 실행", variant="primary").click(
            detect_compare, [cmp_in, conf], [cmp_int8, cmp_fp32, cmp_pt, cmp_table])
        if _sample_images():
            gr.Examples(_sample_images(), cmp_in, label="샘플 이미지")

    with gr.Tab("영상 (CCTV)"):
        with gr.Row():
            vmodel = gr.Dropdown(list(CKPTS.keys()),
                                 value="INT8 (배포형, 3.6MB)", label="모델")
            dev_vid = gr.Dropdown(DEV_CHOICES, value="CPU", label="장치")
        with gr.Row():
            vid_in = gr.Video(label="입력 영상")
            vid_out = gr.Video(label="탐지 결과")
        max_sec = gr.Slider(1, 15, value=5, step=1, label="처리 길이(초)")
        vid_info = gr.Textbox(label="정보", lines=4)
        gr.Button("영상 탐지 실행", variant="primary").click(
            detect_video, [vid_in, vmodel, conf, max_sec, dev_vid], [vid_out, vid_info])
        if _sample_videos():
            gr.Examples(_sample_videos(), vid_in, label="샘플 영상 (교통 CCTV)")

    with gr.Tab("실시간 (웹캠)"):
        gr.Markdown(
            "노트북 웹캠으로 **실시간 탐지**를 확인합니다. (브라우저가 카메라 권한을 요청하면 허용)\n"
            "- 가벼운 **INT8/fp32** 모델 권장 (CPU 실시간)\n"
            "- 처리 FPS는 노트북 CPU 사양에 따라 달라집니다")
        wmodel = gr.Dropdown(list(CKPTS.keys()),
                             value="INT8 (배포형, 3.6MB)", label="모델")
        with gr.Row():
            cam_in = gr.Image(sources=["webcam"], streaming=True, type="numpy",
                              label="웹캠 입력")
            cam_out = gr.Image(label="실시간 탐지 결과")
        cam_info = gr.Textbox(label="실시간 FPS", lines=1)
        cam_in.stream(detect_stream, [cam_in, wmodel, conf], [cam_out, cam_info],
                      stream_every=0.1, concurrency_limit=1)

if __name__ == "__main__":
    port = int(os.environ.get("QYOLO_PORT", "7861"))
    demo.launch(server_name="0.0.0.0", server_port=port,
                share=os.environ.get("QYOLO_SHARE", "0") == "1")
