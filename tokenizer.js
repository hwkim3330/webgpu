// Real working tokenizer for Korean/English
class KoreanEnglishTokenizer {
    constructor() {
        this.vocab = {};
        this.reverseVocab = {};
        this.vocabSize = 32000;
        this.unkToken = '<unk>';
        this.bosToken = '<s>';
        this.eosToken = '</s>';
        this.padToken = '<pad>';
        
        this.buildVocab();
    }

    buildVocab() {
        let id = 0;
        
        // Special tokens
        this.vocab[this.padToken] = id++;
        this.vocab[this.bosToken] = id++;
        this.vocab[this.eosToken] = id++;
        this.vocab[this.unkToken] = id++;
        
        // Common punctuation and symbols
        const punct = '.,!?;:()[]{}"\'-_=+*&^%$#@~`|\\/<>';
        for (const char of punct) {
            this.vocab[char] = id++;
        }
        
        // Numbers
        for (let i = 0; i <= 9; i++) {
            this.vocab[i.toString()] = id++;
        }
        
        // English alphabet (lowercase and uppercase)
        for (let i = 0; i < 26; i++) {
            this.vocab[String.fromCharCode(97 + i)] = id++; // a-z
            this.vocab[String.fromCharCode(65 + i)] = id++; // A-Z
        }
        
        // Space and common whitespace
        this.vocab[' '] = id++;
        this.vocab['\n'] = id++;
        this.vocab['\t'] = id++;
        
        // Common Korean syllables (가-힣)
        const commonKorean = [
            '가', '나', '다', '라', '마', '바', '사', '아', '자', '차', '카', '타', '파', '하',
            '각', '간', '갈', '감', '갑', '강', '개', '객', '거', '건', '걸', '검', '게', '겨',
            '고', '곡', '골', '공', '과', '관', '광', '교', '구', '국', '군', '굴', '궁', '권',
            '그', '극', '근', '글', '금', '기', '긴', '길', '김', '까', '깨', '꺼', '께', '꼬',
            '꽃', '꾸', '꿈', '끝', '나', '낙', '날', '남', '납', '낭', '내', '냉', '너', '넘',
            '네', '녀', '년', '노', '녹', '논', '놀', '농', '누', '눈', '뉴', '느', '늘', '능',
            '니', '다', '단', '달', '담', '답', '당', '대', '댁', '더', '덕', '데', '도', '독',
            '돈', '돌', '동', '되', '두', '둥', '드', '든', '들', '등', '디', '따', '땅', '때',
            '또', '뜨', '뜻', '라', '락', '란', '랄', '람', '랑', '래', '랜', '량', '러', '럭',
            '런', '럴', '레', '려', '련', '례', '로', '록', '론', '롤', '롬', '롱', '료', '루',
            '룩', '룸', '룹', '룽', '르', '른', '를', '름', '릅', '리', '린', '릴', '림', '립',
            '마', '막', '만', '말', '맘', '맞', '망', '매', '맨', '맵', '맹', '머', '먹', '메',
            '멘', '며', '면', '명', '모', '목', '몬', '몰', '몸', '못', '몽', '묘', '무', '묵',
            '문', '물', '뭄', '뭅', '미', '민', '밀', '밉', '바', '박', '반', '발', '밤', '밥',
            '방', '배', '백', '뱀', '버', '번', '벌', '범', '법', '베', '벤', '별', '병', '보',
            '복', '본', '볼', '봄', '봅', '봉', '부', '북', '분', '불', '붐', '붕', '브', '비',
            '빈', '빌', '빔', '빗', '빙', '사', '삭', '산', '살', '삼', '삽', '상', '새', '색',
            '생', '서', '석', '선', '설', '섬', '섭', '성', '세', '센', '셈', '션', '소', '속',
            '손', '솔', '솜', '송', '수', '숙', '순', '술', '숨', '숭', '슈', '스', '슨', '슬',
            '슴', '습', '승', '시', '식', '신', '실', '심', '십', '싱', '싸', '쌀', '쌍', '써',
            '쓰', '씨', '아', '악', '안', '알', '암', '압', '앙', '애', '액', '앤', '야', '약',
            '얀', '얄', '얌', '양', '어', '억', '언', '얼', '엄', '업', '에', '엔', '여', '역',
            '연', '열', '염', '엽', '영', '예', '옛', '오', '옥', '온', '올', '옴', '옵', '와',
            '완', '왈', '왕', '외', '요', '욕', '용', '우', '욱', '운', '울', '움', '웁', '웅',
            '워', '원', '월', '웜', '위', '윈', '유', '육', '윤', '율', '융', '으', '은', '을',
            '음', '응', '의', '이', '익', '인', '일', '임', '입', '잉', '자', '작', '잔', '잘',
            '잠', '잡', '장', '재', '잭', '쟁', '저', '적', '전', '절', '점', '정', '제', '젠',
            '조', '족', '존', '졸', '종', '좋', '주', '죽', '준', '줄', '줌', '줍', '중', '즈',
            '즉', '즐', '즘', '증', '지', '직', '진', '질', '짐', '집', '징', '짜', '째', '쪼',
            '쭈', '차', '착', '찬', '찰', '참', '찹', '창', '채', '책', '챔', '처', '척', '천',
            '철', '첨', '첩', '청', '체', '첸', '초', '촉', '촌', '총', '최', '추', '축', '춘',
            '출', '춤', '춥', '충', '취', '츠', '측', '층', '치', '칙', '친', '칠', '침', '칩',
            '칭', '카', '칵', '칸', '칼', '캄', '캅', '캉', '캐', '캔', '캠', '커', '컨', '컬',
            '컴', '컵', '컹', '케', '켄', '켈', '켐', '코', '콘', '콜', '콤', '콩', '쿠', '쿡',
            '쿨', '쿰', '쿵', '크', '큰', '클', '큼', '큽', '키', '킨', '킬', '킴', '킵', '킹',
            '타', '탁', '탄', '탈', '탐', '탑', '탕', '태', '택', '탬', '터', '턱', '턴', '털',
            '텀', '텁', '테', '텐', '텔', '템', '토', '톤', '톨', '톰', '통', '투', '툰', '툴',
            '툼', '툽', '퉁', '트', '특', '튼', '틀', '틈', '틉', '티', '틴', '틸', '팀', '팁',
            '팅', '파', '팍', '판', '팔', '팜', '팝', '팡', '패', '팩', '팬', '팸', '퍼', '퍽',
            '펀', '펄', '펌', '펍', '페', '펜', '펠', '펨', '펩', '포', '폭', '폰', '폴', '폼',
            '폽', '퐁', '푸', '푹', '푼', '풀', '품', '풍', '프', '픈', '플', '픔', '피', '픽',
            '핀', '필', '핌', '핍', '핑', '하', '학', '한', '할', '함', '합', '항', '해', '핵',
            '핸', '햄', '햅', '행', '허', '헉', '헌', '헐', '험', '헙', '헝', '혀', '현', '혈',
            '혐', '협', '형', '혜', '호', '혹', '혼', '홀', '홈', '홉', '홍', '화', '확', '환',
            '활', '황', '회', '획', '횟', '후', '훅', '훈', '훌', '훔', '훗', '훨', '휘', '휴',
            '흄', '흔', '흘', '흙', '흠', '흡', '흥', '희', '흰', '히', '힌', '힐', '힘', '힙'
        ];
        
        for (const syllable of commonKorean) {
            if (id < this.vocabSize - 1000) { // Leave space for subwords
                this.vocab[syllable] = id++;
            }
        }
        
        // Common English words and Korean words
        const commonWords = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for',
            'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his',
            'by', 'from', 'they', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would',
            'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
            'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just',
            'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some',
            'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only',
            'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
            'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want',
            'because', 'any', 'these', 'give', 'day', 'most', 'us',
            // Korean common words
            '이것', '저것', '그것', '여기', '저기', '거기', '지금', '오늘', '내일', '어제',
            '언제', '어디', '누구', '무엇', '왜', '어떻게', '얼마', '몇', '많이', '조금',
            '크다', '작다', '좋다', '나쁘다', '예쁘다', '못생기다', '빠르다', '느리다',
            '높다', '낮다', '길다', '짧다', '넓다', '좁다', '뜨겁다', '차갑다',
            '먹다', '마시다', '보다', '듣다', '말하다', '읽다', '쓰다', '걷다', '뛰다',
            '앉다', '서다', '누워', '자다', '일어나다', '가다', '오다', '돌아가다',
            '사람', '남자', '여자', '아이', '어른', '친구', '가족', '부모', '형제',
            '학교', '집', '회사', '병원', '상점', '식당', '공원', '도서관',
            '음식', '물', '밥', '빵', '과일', '야채', '고기', '생선', '우유', '커피'
        ];
        
        for (const word of commonWords) {
            if (id < this.vocabSize - 100) { // Leave space for subwords
                this.vocab[word] = id++;
            }
        }
        
        // Fill remaining with random subwords (simplified)
        while (id < this.vocabSize) {
            this.vocab[`<sub_${id}>`] = id++;
        }
        
        // Build reverse vocab
        for (const [token, id] of Object.entries(this.vocab)) {
            this.reverseVocab[id] = token;
        }
    }

    encode(text) {
        if (!text) return [];
        
        // Simple word-level tokenization with fallback to character level
        const tokens = [];
        const words = text.split(/(\s+|[^\w\uAC00-\uD7AF])/);
        
        for (const word of words) {
            if (!word) continue;
            
            if (this.vocab[word] !== undefined) {
                tokens.push(this.vocab[word]);
            } else {
                // Character-level fallback
                for (const char of word) {
                    if (this.vocab[char] !== undefined) {
                        tokens.push(this.vocab[char]);
                    } else {
                        tokens.push(this.vocab[this.unkToken]);
                    }
                }
            }
        }
        
        return tokens;
    }

    decode(tokens) {
        if (!Array.isArray(tokens)) return '';
        
        return tokens
            .map(token => this.reverseVocab[token] || this.unkToken)
            .join('')
            .replace(/\s+/g, ' ')
            .trim();
    }

    getSpecialTokenIds() {
        return {
            pad: this.vocab[this.padToken],
            bos: this.vocab[this.bosToken],
            eos: this.vocab[this.eosToken],
            unk: this.vocab[this.unkToken]
        };
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = KoreanEnglishTokenizer;
}