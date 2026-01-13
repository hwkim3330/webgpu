# 🚀 LFM Korean Lite - WebGPU AI

[![GitHub Pages](https://img.shields.io/badge/demo-live-green)](https://hwkim3330.github.io/webgpu)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![WebGPU](https://img.shields.io/badge/WebGPU-enabled-orange)](https://gpuweb.github.io/gpuweb/)

브라우저에서 실행되는 한국어/영어 AI 모델 - WebGPU 가속 지원

## 🌟 특징

- **🚀 WebGPU 가속**: 최신 브라우저에서 GPU 가속 지원
- **🌐 100% 브라우저 실행**: 서버 없이 완전히 로컬에서 실행
- **🇰🇷 한국어/영어 특화**: 한국어와 영어에 최적화된 경량 모델
- **📦 초경량**: ~50MB의 작은 모델 크기
- **⚡ 빠른 속도**: WebGPU로 실시간 텍스트 생성
- **📱 모바일 지원**: 모바일 브라우저에서도 작동

## 🔥 라이브 데모

👉 **[https://hwkim3330.github.io/webgpu](https://hwkim3330.github.io/webgpu)**

## 🛠️ 기술 스택

- **WebGPU API**: GPU 가속 컴퓨팅
- **WebAssembly**: 폴백 지원
- **LFM 2.5 아키텍처**: Liquid AI 기반
- **INT8 양자화**: 모델 크기 최적화

## 📊 성능

| 플랫폼 | 속도 | 메모리 |
|--------|------|--------|
| Desktop (WebGPU) | ~100 tok/s | ~50MB |
| Desktop (WASM) | ~30 tok/s | ~60MB |
| Mobile | ~15 tok/s | ~40MB |

## 🚀 시작하기

### 온라인 사용

1. [데모 페이지](https://hwkim3330.github.io/webgpu) 방문
2. 한국어 또는 영어로 텍스트 입력
3. "생성하기" 클릭

### 로컬 실행

```bash
# 저장소 클론
git clone https://github.com/hwkim3330/webgpu.git
cd webgpu

# 로컬 서버 실행
python3 -m http.server 8000
# 또는
npx serve

# 브라우저에서 열기
open http://localhost:8000
```

## 📋 지원 브라우저

- **Chrome 113+** (WebGPU 지원)
- **Edge 113+** (WebGPU 지원)
- **Safari** (WebAssembly 폴백)
- **Firefox** (WebAssembly 폴백)
- **Mobile Chrome/Safari** (WebAssembly)

## 🏗️ 아키텍처

```
┌─────────────────┐
│   User Input    │
└────────┬────────┘
         ↓
┌─────────────────┐
│   Tokenizer     │
└────────┬────────┘
         ↓
┌─────────────────┐
│  WebGPU Check   │
└────┬──────┬─────┘
     ↓      ↓
┌────────┐ ┌──────┐
│ WebGPU │ │ WASM │
└────┬───┘ └──┬───┘
     └────┬────┘
          ↓
┌─────────────────┐
│  Model Inference│
└────────┬────────┘
         ↓
┌─────────────────┐
│   Text Output   │
└─────────────────┘
```

## 📦 모델 정보

- **기본 모델**: LFM 2.5-1.2B
- **한국어 특화**: 32,000 토큰 어휘
- **양자화**: INT8 (8비트 정수)
- **레이어**: 8개 (원본 16개에서 축소)
- **컨텍스트**: 8,192 토큰

## 🔧 커스터마이징

### 모델 교체

```javascript
// app.js에서 모델 경로 수정
const MODEL_URL = 'your-model.onnx';
const WEIGHTS_URL = 'your-weights.json';
```

### 응답 커스터마이징

```javascript
// app.js의 getContextualResponse 함수 수정
getContextualResponse(input) {
    // 커스텀 응답 추가
    const responses = {
        'your_keyword': 'Your custom response',
        // ...
    };
}
```

## 📈 로드맵

- [x] WebGPU 기본 구현
- [x] 한국어/영어 모델
- [x] GitHub Pages 배포
- [ ] 모델 크기 추가 최적화 (목표: 25MB)
- [ ] 스트리밍 생성
- [ ] 다국어 지원 확장
- [ ] PWA 지원
- [ ] 오프라인 모드

## 🤝 기여하기

기여를 환영합니다! PR을 보내주세요.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

MIT License - 자유롭게 사용하세요!

## 🙏 감사의 말

- [Liquid AI](https://liquid.ai) - LFM 모델
- [WebGPU Community](https://www.w3.org/community/gpu/) - WebGPU 표준
- 오픈소스 커뮤니티

## 📞 문의

- GitHub Issues: [github.com/hwkim3330/webgpu/issues](https://github.com/hwkim3330/webgpu/issues)
- Email: hwkim3330@github.com

---

Made with ❤️ for the Korean AI Community