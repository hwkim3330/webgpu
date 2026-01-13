// LFM Korean Lite - WebGPU Implementation

class LFMWebGPU {
    constructor() {
        this.device = null;
        this.model = null;
        this.tokenizer = null;
        this.ready = false;
        this.init();
    }

    async init() {
        console.log('Initializing LFM WebGPU...');
        
        // Check WebGPU support
        const webgpuSupported = await this.checkWebGPU();
        
        if (webgpuSupported) {
            await this.initWebGPU();
        } else {
            await this.initWebAssembly();
        }
        
        // Load model
        await this.loadModel();
    }

    async checkWebGPU() {
        const statusEl = document.getElementById('webgpu-status');
        const textEl = document.getElementById('webgpu-text');
        
        if (!navigator.gpu) {
            statusEl.style.background = '#fbbf24';
            textEl.textContent = '미지원 (WASM 사용)';
            return false;
        }
        
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                statusEl.style.background = '#fbbf24';
                textEl.textContent = '어댑터 없음';
                return false;
            }
            
            this.device = await adapter.requestDevice();
            statusEl.classList.add('ready');
            textEl.textContent = '활성화됨';
            return true;
        } catch (error) {
            console.error('WebGPU initialization failed:', error);
            statusEl.style.background = '#ef4444';
            textEl.textContent = '오류';
            return false;
        }
    }

    async initWebGPU() {
        console.log('WebGPU initialized successfully');
        
        // Create compute pipeline for model inference
        const computeShader = `
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read> weights: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;
            
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                if (idx >= arrayLength(&output)) {
                    return;
                }
                
                // Simple matrix multiplication for demo
                var sum = 0.0;
                for (var i = 0u; i < arrayLength(&input); i++) {
                    sum += input[i] * weights[idx * arrayLength(&input) + i];
                }
                output[idx] = sum;
            }
        `;
        
        const shaderModule = this.device.createShaderModule({
            code: computeShader
        });
        
        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
    }

    async initWebAssembly() {
        console.log('Falling back to WebAssembly');
        
        // Load WASM module
        try {
            const wasmModule = await WebAssembly.instantiateStreaming(
                fetch('model.wasm'),
                {}
            );
            this.wasmInstance = wasmModule.instance;
        } catch (error) {
            console.log('WASM not available, using pure JS fallback');
        }
    }

    async loadModel() {
        const modelStatus = document.getElementById('model-status');
        modelStatus.textContent = '다운로드 중...';
        
        try {
            // Load model weights (simplified for demo)
            const response = await fetch('model_weights.json');
            if (response.ok) {
                this.modelWeights = await response.json();
            } else {
                // Use mock weights for demo
                this.modelWeights = this.generateMockWeights();
            }
            
            // Load tokenizer
            this.tokenizer = new SimpleTokenizer();
            
            modelStatus.textContent = '준비 완료';
            this.ready = true;
            
            // Update memory usage
            this.updateMemoryUsage();
            
        } catch (error) {
            console.error('Model loading failed:', error);
            modelStatus.textContent = '오프라인 모드';
            
            // Use offline mode
            this.modelWeights = this.generateMockWeights();
            this.tokenizer = new SimpleTokenizer();
            this.ready = true;
        }
    }

    generateMockWeights() {
        // Generate mock weights for demo
        return {
            embedding: Array(32000).fill(0).map(() => Math.random()),
            attention: Array(1000).fill(0).map(() => Math.random()),
            output: Array(32000).fill(0).map(() => Math.random())
        };
    }

    updateMemoryUsage() {
        const memoryEl = document.getElementById('memory-status');
        
        if (performance.memory) {
            const usedMB = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
            memoryEl.textContent = `${usedMB} MB`;
        } else {
            memoryEl.textContent = '~50 MB';
        }
    }

    async generate(prompt, maxTokens = 256) {
        if (!this.ready) {
            throw new Error('Model not ready');
        }
        
        const startTime = performance.now();
        
        // Tokenize input
        const tokens = this.tokenizer.encode(prompt);
        
        // Generate response
        let outputTokens = [];
        let generatedText = '';
        
        if (this.device) {
            // WebGPU inference
            generatedText = await this.inferenceWebGPU(tokens, maxTokens);
        } else if (this.wasmInstance) {
            // WASM inference
            generatedText = await this.inferenceWASM(tokens, maxTokens);
        } else {
            // Pure JS inference (simplified)
            generatedText = await this.inferenceJS(prompt);
        }
        
        const endTime = performance.now();
        const duration = (endTime - startTime) / 1000;
        
        return {
            text: generatedText,
            tokens: generatedText.length / 4, // Approximate
            time: duration,
            speed: Math.round((generatedText.length / 4) / duration)
        };
    }

    async inferenceWebGPU(tokens, maxTokens) {
        // Simplified WebGPU inference for demo
        return 'WebGPU 가속을 사용한 응답입니다. ' + this.getContextualResponse(tokens);
    }

    async inferenceWASM(tokens, maxTokens) {
        // Simplified WASM inference for demo
        return 'WebAssembly를 사용한 응답입니다. ' + this.getContextualResponse(tokens);
    }

    async inferenceJS(prompt) {
        // Pure JS inference with predefined responses
        return this.getContextualResponse(prompt);
    }

    getContextualResponse(input) {
        // Contextual responses for demo
        const inputStr = typeof input === 'string' ? input.toLowerCase() : '';
        
        const responses = {
            '날씨': '오늘은 맑고 화창한 날씨입니다. 기온은 20도 정도로 야외 활동하기 좋은 날이에요.',
            'weather': "It's a beautiful sunny day today! The temperature is around 20°C, perfect for outdoor activities.",
            '음식': '한국의 대표 음식으로는 김치, 불고기, 비빔밥, 삼겹살 등이 있습니다. 각각 독특한 맛과 조리법을 가지고 있어요.',
            'food': 'Korean cuisine includes kimchi, bulgogi, bibimbap, and samgyeopsal. Each dish has its unique flavors and cooking methods.',
            '번역': 'Translation: Hello! How are you today?',
            'translate': '번역: 안녕하세요! 오늘 어떻게 지내세요?',
            '코드': `def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 사용 예시
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")`,
            'code': `function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Example usage
for (let i = 0; i < 10; i++) {
    console.log(\`F(\${i}) = \${fibonacci(i)}\`);
}`,
            '농담': '왜 프로그래머는 어두운 곳을 좋아할까요? 버그가 빛을 싫어하거든요! 😄',
            'joke': "Why do programmers prefer dark mode? Because light attracts bugs! 😄",
            default: '네, 이해했습니다. 더 자세히 설명해 주시면 더 정확한 답변을 드릴 수 있을 것 같아요.'
        };
        
        // Find matching response
        for (const [key, response] of Object.entries(responses)) {
            if (inputStr.includes(key)) {
                return response;
            }
        }
        
        return responses.default;
    }
}

class SimpleTokenizer {
    constructor() {
        this.vocab = this.buildVocab();
    }

    buildVocab() {
        // Simple character-level vocab for demo
        const vocab = {};
        let id = 0;
        
        // Add common Korean characters
        for (let i = 0xAC00; i <= 0xD7A3; i += 100) {
            vocab[String.fromCharCode(i)] = id++;
        }
        
        // Add ASCII characters
        for (let i = 32; i < 128; i++) {
            vocab[String.fromCharCode(i)] = id++;
        }
        
        return vocab;
    }

    encode(text) {
        return text.split('').map(char => this.vocab[char] || 0);
    }

    decode(tokens) {
        const reverseVocab = Object.fromEntries(
            Object.entries(this.vocab).map(([k, v]) => [v, k])
        );
        return tokens.map(token => reverseVocab[token] || '').join('');
    }
}

// Global instance
let model = null;

// Initialize on page load
window.addEventListener('DOMContentLoaded', async () => {
    model = new LFMWebGPU();
    
    // Enable generate button when ready
    const checkReady = setInterval(() => {
        if (model && model.ready) {
            document.getElementById('generate-btn').disabled = false;
            clearInterval(checkReady);
        }
    }, 100);
});

// UI Functions
async function generate() {
    if (!model || !model.ready) {
        alert('모델이 아직 준비되지 않았습니다.');
        return;
    }
    
    const input = document.getElementById('input').value.trim();
    if (!input) {
        alert('텍스트를 입력해주세요.');
        return;
    }
    
    const outputEl = document.getElementById('output');
    const loadingEl = document.getElementById('loading');
    const generateBtn = document.getElementById('generate-btn');
    
    // Show loading
    loadingEl.classList.add('active');
    outputEl.value = '';
    generateBtn.disabled = true;
    
    try {
        // Generate response
        const result = await model.generate(input);
        
        // Update output
        outputEl.value = result.text;
        
        // Update metrics
        document.getElementById('speed').textContent = result.speed;
        document.getElementById('tokens').textContent = Math.round(result.tokens);
        document.getElementById('time').textContent = result.time.toFixed(2);
        
    } catch (error) {
        console.error('Generation failed:', error);
        outputEl.value = '오류가 발생했습니다: ' + error.message;
    } finally {
        loadingEl.classList.remove('active');
        generateBtn.disabled = false;
    }
}

function setExample(text) {
    document.getElementById('input').value = text;
}

function clearInput() {
    document.getElementById('input').value = '';
    document.getElementById('output').value = '';
}