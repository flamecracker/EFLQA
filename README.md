EFLQA Readme.md
===============
프로젝트 개요
-------------
본 프로젝트는 Gutenberg 프로젝트의 동화책 데이터를 활용하여, 자동으로 질문-답변 쌍을 생성하고, 이를 평가하며, 주제별로 클러스터링하는 작업을 수행합니다. 다음은 프로젝트의 주요 컴포넌트 설명입니다.

스크립트
-------------
### 1. my_gutenberg_download.py

용도: Gutenberg 프로젝트에서 받아온 동화책 데이터를 data/book 경로에 저장합니다. 이후 각 책을 여러 개의 문단으로 나누어 data/passage에 저장합니다.

주요 기능: 동화책 데이터의 분할 및 정리
### 2. generate_QA.py

용도: my_gutenberg_download.py에서 생성된 문단들을 사용하여, OpenAI API를 통해 문단을 다듬고, 해당 지문에 대한 최대 6개의 질문-답변 쌍을 생성합니다.

주요 기능: 질문-답변 쌍의 자동 생성
### 3. evaluate_llm.py

용도: 생성된 질문-답변 데이터의 질을 평가하기 위해 Llama2 모델을 사용합니다.

주요 기능: 데이터의 품질 평가
### 4. topic_modeling.py

용도: 책의 내용을 입력받아 약 30개의 토픽으로 클러스터링합니다.

주요 기능: 주제별 클러스터링
### 5. topic_title_match.py

용도: 각 토픽에 해당하는 책의 이름과 저자를 연결하여 시각화합니다.

주요 기능: 결과 시각화

데이터 파일들
-------------
본 프로젝트를 통해 편집 및 생성된 데이터 파일들입니다. GitHub 웹 인터페이스에서는 최대 1,000개의 파일만 표시되지만, 다운로드 시에는 약 1,570개의 데이터 파일이 온전히 포함되어 있습니다. 이 데이터들은 프로젝트의 스크립트를 통해 생성되며, 프로젝트의 핵심 부분을 이룹니다.
사용 방법
프로젝트를 로컬 시스템에 클론한 후, 위의 스크립트들을 순서대로 실행하면 됩니다. 각 스크립트는 특정한 목적에 맞게 데이터를 처리하고 결과를 출력합니다.

주의 사항
-------------
본 프로젝트는 OpenAI API와 Llama2 모델을 사용합니다. 해당 API와 모델을 사용하기 위한 준비가 필요합니다.
데이터 파일들은 크기가 크므로, GitHub에서 직접 다운로드하는 대신, Git 클론을 권장합니다.
이 프로젝트는 Gutenberg 프로젝트의 공개 데이터를 활용하여, 자연어 처리와 기계 학습 기술을 이용한 데이터 처리 및 분석의 예시를 보여줍니다.
