// Korean-English conversation dataset for better responses
const koreanEnglishDataset = {
    // 인사 및 일상 대화
    greetings: {
        '안녕하세요': ['안녕하세요! 반가워요. 오늘 하루는 어떠셨나요?', '안녕하세요! 무엇을 도와드릴까요?', '안녕하세요! 좋은 하루 되세요!'],
        '안녕': ['안녕하세요! 편하게 말씀하세요.', '안녕! 뭘 도와드릴까요?', '안녕하세요! 어떤 이야기를 나누고 싶으신가요?'],
        'hello': ['Hello! How are you doing today?', 'Hello! What can I help you with?', 'Hello! Nice to meet you!'],
        'hi': ['Hi there! How can I assist you?', 'Hi! What brings you here today?', 'Hi! Let\'s have a great conversation!'],
        '좋은 아침': ['좋은 아침입니다! 오늘도 활기찬 하루 되세요!', '아침 인사 감사해요! 커피는 드셨나요?'],
        'good morning': ['Good morning! Hope you have a wonderful day ahead!', 'Good morning! Ready to start the day?']
    },

    // 날씨 관련
    weather: {
        '날씨': [
            '오늘 날씨는 맑고 화창합니다! 기온은 22도 정도로 야외 활동하기 딱 좋은 날씨예요. 산책이나 운동하기에 완벽한 조건이네요.',
            '요즘 날씨가 참 좋죠? 봄바람이 살랑살랑 불어서 기분이 좋아져요. 창문을 열어두시면 자연의 향기를 느낄 수 있을 거예요.',
            '날씨에 대해 궁금하시군요! 지금은 구름이 조금 있지만 전체적으로 맑은 편이에요. 우산은 필요 없을 것 같아요.'
        ],
        'weather': [
            'The weather today is sunny and clear! Temperature is around 22°C, perfect for outdoor activities. Great conditions for a walk or exercise.',
            'The weather has been really nice lately! There\'s a gentle spring breeze that makes you feel good. Opening windows lets you enjoy the natural fragrance.',
            'Curious about the weather? It\'s mostly clear with some light clouds. You probably won\'t need an umbrella today.'
        ],
        '비': ['비가 오는 날엔 실내에서 독서나 영화 감상은 어떨까요? 비 소리를 들으며 따뜻한 차 한 잔도 좋겠어요.'],
        'rain': ['On rainy days, how about reading a book or watching a movie indoors? A warm cup of tea while listening to the rain sounds perfect.']
    },

    // 음식 관련
    food: {
        '음식': [
            '한국 음식 중에서 특히 추천하고 싶은 건 김치찌개예요! 김치의 신맛과 돼지고기의 고소함이 어우러져 정말 맛있어요. 밥 한 그릇이 뚝딱 사라지죠.',
            '불고기는 한국을 대표하는 음식 중 하나예요. 달콤한 양념에 재운 소고기를 구워서 먹는데, 외국인들도 정말 좋아해요. 상추에 싸서 먹으면 더욱 맛있어요!',
            '비빔밥 어떠세요? 다양한 나물과 고기, 달걀이 올라간 영양 만점 요리예요. 고추장과 잘 비벼서 먹으면 건강하고 맛있어요.'
        ],
        'food': [
            'Korean food is amazing! I especially recommend kimchi jjigae (kimchi stew). The sour taste of kimchi combined with the rich flavor of pork is absolutely delicious.',
            'Bulgogi is one of Korea\'s representative dishes. Sweet marinated beef that\'s grilled to perfection. Even foreigners love it! It tastes even better wrapped in lettuce.',
            'How about bibimbap? It\'s a nutritious dish with various vegetables, meat, and egg on rice. Mix it well with gochujang (red pepper paste) for a healthy and delicious meal.'
        ],
        '커피': [
            '커피 좋아하시는군요! 요즘 드립 커피가 인기가 많죠. 원두의 향과 맛을 제대로 느낄 수 있어서 좋아요. 어떤 원두를 선호하시나요?',
            '카페에서 마시는 커피도 좋지만, 집에서 핸드드립으로 내려마시는 커피도 특별한 맛이 있어요. 천천히 내리는 과정 자체가 힐링이 되죠.'
        ],
        'coffee': [
            'I see you like coffee! Drip coffee is really popular these days. You can truly taste and smell the beans. What kind of beans do you prefer?',
            'Coffee from cafes is nice, but hand-drip coffee at home has a special taste. The slow brewing process itself is quite therapeutic.'
        ]
    },

    // 취미 및 관심사
    hobbies: {
        '음악': [
            '음악 좋아하시는군요! 어떤 장르를 즐겨 듣나요? K-pop, 클래식, 재즈 등 다양한 장르가 있는데, 각각의 매력이 달라요.',
            '음악은 정말 좋은 취미죠! 스트레스 해소에도 도움이 되고, 기분도 좋아져요. 혹시 악기도 다루시나요?'
        ],
        'music': [
            'I see you like music! What genre do you enjoy? K-pop, classical, jazz - each has its own unique charm.',
            'Music is such a great hobby! It helps relieve stress and improves your mood. Do you play any instruments?'
        ],
        '책': [
            '독서는 정말 좋은 습관이에요! 어떤 종류의 책을 읽으시나요? 소설, 에세이, 자기계발서 등 다양한 장르가 있죠.',
            '책을 통해 새로운 세상을 경험할 수 있어서 좋아요. 최근에 읽은 인상 깊은 책이 있으신가요?'
        ],
        'book': [
            'Reading is such a wonderful habit! What kind of books do you read? Novels, essays, self-help books - there are so many genres.',
            'Books let you experience new worlds. Have you read any impressive books recently?'
        ]
    },

    // 기술 및 프로그래밍
    technology: {
        '프로그래밍': [
            '프로그래밍에 관심이 있으시군요! 어떤 언어를 배우고 계신가요? Python, JavaScript, Java 등 각각의 특징이 달라요.',
            '코딩은 논리적 사고력을 기르는 데 도움이 되죠. 처음에는 어려울 수 있지만, 차근차근 배우시면 분명 재미있어질 거예요!'
        ],
        'programming': [
            'Interested in programming! What language are you learning? Python, JavaScript, Java - each has its own characteristics.',
            'Coding helps develop logical thinking. It might be difficult at first, but if you learn step by step, it will definitely become fun!'
        ],
        'AI': [
            'AI 기술이 정말 빠르게 발전하고 있어요! 일상생활에서도 점점 더 많이 활용되고 있죠. 어떤 부분이 가장 관심 있으신가요?',
            '인공지능은 미래 사회를 바꿀 혁신적인 기술이에요. 하지만 윤리적인 고려사항도 함께 생각해야 할 중요한 주제죠.'
        ],
        '인공지능': [
            'AI 기술의 발전이 정말 놀라워요! 번역, 이미지 생성, 텍스트 작성 등 다양한 분야에서 활용되고 있어요.',
            '인공지능 시대에는 창의성과 감정적 지능이 더욱 중요해질 것 같아요. 기술과 인간이 조화롭게 발전해야겠죠.'
        ]
    },

    // 학습 및 교육
    learning: {
        '공부': [
            '공부하고 계시는군요! 어떤 분야를 공부하고 계신가요? 집중력을 높이려면 25분 공부 후 5분 휴식하는 포모도로 기법을 추천해요.',
            '꾸준한 학습이 가장 중요해요. 매일 조금씩이라도 지속하는 것이 큰 성과를 만들어낼 거예요. 화이팅!'
        ],
        'study': [
            'Great that you\'re studying! What field are you focusing on? For better concentration, I recommend the Pomodoro technique - 25 minutes of study followed by 5-minute breaks.',
            'Consistent learning is the most important thing. Even studying a little bit every day will lead to great results. Keep it up!'
        ],
        '영어': [
            '영어 공부하시는군요! 영어 실력 향상에는 꾸준한 연습이 중요해요. 영화나 드라마를 영어 자막으로 보는 것도 도움이 돼요.',
            '영어 회화 실력을 늘리고 싶으시다면, 혼자서라도 소리내서 연습해보세요. 거울을 보며 말하는 연습도 효과적이에요!'
        ],
        'english': [
            'Studying English! Consistent practice is key to improving English skills. Watching movies or dramas with English subtitles can be really helpful.',
            'If you want to improve your English conversation skills, practice speaking out loud even when alone. Practicing in front of a mirror is also effective!'
        ]
    },

    // 감정 및 위로
    emotions: {
        '힘들어': [
            '힘든 시간을 보내고 계시는군요. 모든 사람에게는 어려운 순간이 있어요. 잠시 깊게 숨을 쉬고, 자신에게 따뜻한 말을 건네보세요.',
            '때로는 힘들 때가 있죠. 그럴 때는 좋아하는 음악을 듣거나, 따뜻한 차를 마시며 마음을 달래보세요. 이 또한 지나갈 거예요.'
        ],
        '기쁘다': [
            '기쁜 일이 있으셨나보네요! 좋은 소식이 있으시면 언제든 공유해주세요. 함께 기뻐하고 싶어요!',
            '행복한 감정이 느껴져서 저도 덩달아 기분이 좋아져요. 오늘의 좋은 기운이 계속 이어지길 바라요!'
        ],
        'tired': [
            'Feeling tired? Everyone needs rest sometimes. How about taking a short break, listening to some calm music, or having a warm drink?',
            'When you\'re tired, don\'t push yourself too hard. Give yourself permission to rest and recharge. You deserve it!'
        ],
        'happy': [
            'I can sense your happiness! If you have good news, feel free to share it anytime. I\'d love to celebrate with you!',
            'Your happy emotions are contagious - they make me feel good too! I hope this positive energy continues throughout your day!'
        ]
    },

    // 번역 관련
    translation: {
        'translate': {
            '안녕하세요': 'Hello',
            '감사합니다': 'Thank you',
            '미안해요': 'I\'m sorry',
            '사랑해': 'I love you',
            '잘 자요': 'Good night',
            '맛있어요': 'It\'s delicious',
            '예쁘다': 'Pretty/Beautiful',
            '재미있어요': 'It\'s fun/interesting',
            '행복해요': 'I\'m happy',
            '건강하세요': 'Stay healthy'
        },
        '번역': {
            'hello': '안녕하세요',
            'thank you': '감사합니다',
            'sorry': '미안해요',
            'love you': '사랑해요',
            'good night': '잘 자요',
            'delicious': '맛있어요',
            'beautiful': '예뻐요',
            'interesting': '재미있어요',
            'happy': '행복해요',
            'healthy': '건강해요'
        }
    }
};

// Function to get contextual response
function getContextualResponse(input) {
    const inputLower = input.toLowerCase().trim();
    
    // Check each category
    for (const [category, data] of Object.entries(koreanEnglishDataset)) {
        for (const [key, responses] of Object.entries(data)) {
            if (inputLower.includes(key.toLowerCase())) {
                if (Array.isArray(responses)) {
                    return responses[Math.floor(Math.random() * responses.length)];
                } else if (typeof responses === 'object') {
                    // Handle translation objects
                    for (const [phrase, translation] of Object.entries(responses)) {
                        if (inputLower.includes(phrase.toLowerCase())) {
                            return `"${phrase}" → "${translation}"`;
                        }
                    }
                }
            }
        }
    }
    
    // Default responses if no match found
    const defaultResponses = [
        '네, 이해했습니다. 더 구체적으로 말씀해 주시면 더 정확한 답변을 드릴 수 있을 것 같아요.',
        '흥미로운 질문이네요! 이에 대해 더 자세히 알아보겠습니다.',
        '좋은 지적이세요. 이 주제에 대해 함께 생각해보면 좋을 것 같습니다.',
        'I understand. Could you provide more details so I can give you a more accurate answer?',
        'That\'s an interesting question! Let me think about this more carefully.',
        'Good point! This is definitely worth exploring further.'
    ];
    
    return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
}

// Export for use in other scripts
if (typeof window !== 'undefined') {
    window.getContextualResponse = getContextualResponse;
    window.koreanEnglishDataset = koreanEnglishDataset;
}