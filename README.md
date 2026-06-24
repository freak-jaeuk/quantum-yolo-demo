# 🔬 Quantum-YOLO 엣지 탐지 데모

YOLO26n 객체탐지기의 어텐션 블록을 **변분 양자 회로(VQC)** 로 학습한 뒤, 그 회로를
**고전 연산으로 무손실 컴파일**하고 **ONNX / INT8** 로 export 한 *실제 배포 아티팩트* 를
어느 노트북에서나 단독 실행할 수 있는 데모입니다. **GPU 불필요, COCO 데이터셋 불필요.**

> **정직한 위치(읽어주세요).** 이 양자 회로(4큐비트)는 고전적으로 시뮬레이션 가능하므로
> *탐지 정확도에서 양자 우위를 주장하지 않습니다* — 통제군 대비 **패리티**임을 확인했습니다.
> 이 데모의 기여는 **"genuine 양자 회로 → 0-오버헤드 고전 컴파일 → INT8 엣지 배포 → 실제
> IBM QPU 검증"이라는 작동하는 파이프라인**입니다. 실세계 가치(서버 GPU 의존도 감소)는
> 양자가 아니라 **고전 엣지 최적화(컴파일·INT8)** 에서 나옵니다.

---

## 빠른 시작 (다른 노트북에서)

```bash
git clone <THIS_REPO_URL> quantum-yolo-demo
cd quantum-yolo-demo

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app.py
# 브라우저에서 http://localhost:7861 열기
```

처음 실행 시 의존성 설치(torch 등)로 몇 분 걸릴 수 있습니다. 이후엔 즉시 뜹니다.

---

## 탭 구성

| 탭 | 설명 |
|---|---|
| **이미지** | 한 이미지에 한 배포형태(INT8/fp32/VQC) 탐지 |
| **비교** | 같은 이미지에 세 배포형태 동시 실행 → INT8(3.6MB)이 fp32와 거의 동일 탐지 확인 |
| **영상 (CCTV)** | 업로드한 영상을 프레임별 탐지 (샘플 교통영상 포함) → 처리 FPS 표시 |
| **실시간 (웹캠)** | **노트북 웹캠 실시간 탐지** — 브라우저 카메라 권한 허용 후 라이브 스트림 |

> 실시간(real-time) 확인: **실시간(웹캠) 탭**에서 라이브로, 또는 **영상 탭**에서 샘플 영상의
> 처리 FPS로 확인할 수 있습니다.

---

## 모델 사실 (풀COCO 학습, 검증된 수치)

YOLO26n + `QuantumC2PSA`(VQC 채널 게이트), COCO `train2017` 118k장 / 80클래스 학습,
epoch 40 수렴. full-val(5000장) 측정:

| 배포 형태 | 크기 | mAP50-95 | FPS (참고: Xeon CPU) |
|---|---|---|---|
| GPU (.pt 컴파일) | — | 0.3635 | **189** |
| CPU fp32 (컴파일 ONNX) | 9.98 MB | **0.3635** | **63.3** |
| CPU INT8 (배포형) | **3.58 MB** | 0.3594 | 21.2 |

- INT8 정확도 손실 **−1.13%**, 크기 **2.8배↓**.
- **정직성:** 이 CPU(ONNXRuntime QDQ)에선 INT8이 fp32보다 *느립니다*(Q/DQ 오버헤드 +
  VNNI int8 커널 부재). **INT8의 이득은 속도가 아니라 크기**(온카메라 탑재). 속도 이득은
  OpenVINO / VNNI / NPU 런타임에서 발현됩니다.
- 위 FPS는 서버 Xeon 기준이며, **노트북 CPU에서는 사양에 따라 달라집니다.**

---

## 폴더 구조

```
quantum-yolo-demo/
├─ app.py                  # gradio 데모 (단독 실행)
├─ requirements.txt
├─ yolo26q/                # QuantumC2PSA 등록 (ultralytics 패치)
│   ├─ register.py
│   └─ bottleneck_vqc.py
├─ models/                 # VQC 레이어 + 고전 컴파일
│   ├─ quantum_layers.py
│   └─ vqc_deploy.py
├─ weights/                # 배포 아티팩트 (풀COCO f100)
│   ├─ best.pt             # PyTorch VQC
│   ├─ best.onnx           # fp32 컴파일 ONNX
│   └─ best_int8_pctA.onnx # INT8 (head fp32 보호)
└─ samples/                # 데모용 샘플 (COCO 데이터셋 불필요)
    ├─ images/
    └─ video/
```

## GPU 사용 (선택)

기본은 CPU입니다. Linux + NVIDIA 드라이버가 있으면:

```bash
LD_LIBRARY_PATH=/usr/lib64 python app.py    # .pt(VQC) 형태가 GPU 사용
```

ONNX 형태는 onnxruntime CPU 빌드라 CPU로 동작합니다(엣지 현실성).

## 라이선스 / 출처

연구용 데모. YOLO26/ultralytics는 각 라이선스를 따릅니다.
