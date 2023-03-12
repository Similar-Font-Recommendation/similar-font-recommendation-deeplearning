# -*- coding: utf-8 -*- 

### 1. 라이브러리 로드
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import sys
sys.path.append("./img2vec_pytorch2")  # Adds higher directory to python modules path.
from img_to_vec import Img2Vec
from PIL import Image, ImageDraw, ImageFont, ImageChops
import os
import cv2
img2vec =Img2Vec(model="inception")

### 2. 텍스트 렌더링
def make_image(message, font_list, img_save_path):

  # font setting
  font_color = 'rgb(0, 0, 0)'
  font_size = 60
  
  # image size
  width = len(message) * font_size + 10
  height = font_size + 35
  bg_color = 'rgb(255, 255, 255)'
  x_margin = 5
  y_margin = 5
  
  # 렌더링 결과 파일 저장할 폴더 생성
  img_save_path = img_save_path + message + '/'
  os.mkdir(img_save_path)
  
  for font in font_list:
    image =Image.new('RGB', (width, height), color = bg_color)
    font_name = font.replace('.ttf', '').replace('.TTF', '')
    font = ImageFont.truetype('./ttf_font(430)/' + font, font_size)
    draw = ImageDraw.Draw(image)      
    draw.text((x_margin, y_margin), message, font=font, fill=font_color)
    
    # save file
    image.save(img_save_path + font_name + '.png')
  
  return img_save_path

#image resize
def resize_img(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    h,w= img.shape
    if h > 60:
        ratio_h = h / 60
        img_resize = cv2.resize(img, None, fx=ratio_h, fy=ratio_h, interpolation = cv2.INTER_CUBIC)
    elif h < 60:
        ratio_h =  60 / h
        img_resize = cv2.resize(img, None, fx=ratio_h, fy=ratio_h, interpolation = cv2.INTER_CUBIC)
    else:
        img_resize = img
    cv2.imwrite(path, img_resize)

### 2-1. 이미지 여백 제거
def trim(img_path):
  img_path = img_path
  img_list = os.listdir(img_path)
  
  for img_name in img_list:
    img = Image.open(img_path + img_name)
    background = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, background)
    diff = ImageChops.add(diff, diff, 2.0, -35)
    bbox = diff.getbbox()
    if bbox:
      img = img.crop(bbox)
      img.save(img_path + img_name)
      resize_img(img_path + img_name)
    else:
      print('Trim Failure!')
      return
    
### 3. input 이미지 특징벡터 추출
input_img_path = "/FontSearching/input_img/"
input_img = 'test.jpg'
test_img = Image.open(os.path.join(input_img_path, input_img)).convert('RGB')
resize_img(input_img_path + input_img)
test_img_vector = img2vec.get_vec(test_img)
test_img_vector_df = pd.DataFrame(test_img_vector).transpose()

### 4. 유사도 비교
#### 4-1. 보유 폰트로 텍스트 렌더링
font_list = os.listdir('./ttf_font(430)')
font_list = sorted(font_list)
msg = "물고기"
img_save_path = "/FontSearching/rendering_result/"
trim_img_path = make_image(msg, font_list, img_save_path)
trim(trim_img_path)


#### 4-2. 렌더링 이미지의 특징벡터 추출
input_path = trim_img_path
list_pics1 = []
filenames1 = []
for file1 in sorted(os.listdir(input_path)):
    filename1 = os.fsdecode(file1)
    img1 = Image.open(os.path.join(input_path, filename1)).convert('RGB')
    list_pics1.append(img1)
    filenames1.append(filename1)

vectors1 = img2vec.get_vec(list_pics1)
pics1 = {}
for i1, vec1 in enumerate(vectors1):
    pics1[filenames1[i1]] = vec1
    
rendering_vector_df = pd.DataFrame(pics1).transpose()
rendering_vector_df = rendering_vector_df.sort_index(ascending=True)

#### 4-3. input 이미지 ↔ 렌더링 이미지의 유사도 비교
rendering_vec_sim = []
for i in range(len(rendering_vector_df)):
  rendering_vec_sim.append(cosine_similarity(test_img_vector_df.values, rendering_vector_df.iloc[[i]].values)[0][0])

rendering_vector_sim_df = rendering_vector_df
rendering_vector_sim_df['similarity'] = rendering_vec_sim

linux_fontname = ["텐바이텐", "텐바이텐 Bold", "116앵무부리", "116수박화체", "12롯데마트드림Bold", "12롯데마트드림Light", "12롯데마트드림Medium", "12롯데마트행복Bold", "12롯데마트행복Light", "12롯데마트행복Medium", "도서관체", "성동고딕", "성동고딕B", "경기천년제목 Bold", "경기천년제목 Medium", "경기천년제목V Bold", "경기천년바탕 Bold", "경기천년바탕 Regular", "빛고을광주체 Light", "빛고을광주체 Medium", "빛고을광주체 Bold", "김포평화바탕", "순천체R", "순천체B", "전주완판본 순체 B", "전주완판본 순체 L", "전주완판본 순체 R", "전주완판본 각체 B", "전주완판본 각체 L", "전주완판본 각체 R", "유토이미지 고딕 R", "유토이미지 별나라달님체", "유토이미지 빨간우체통체", "유토이미지 플라워체", "유토이미지체", "유토이미지 고딕 B", "유토이미지 고딕 L", "아리따 부리 B", "아리따 부리 L", "아리따 부리 M", "아리따 부리 SB", "애터미체 Bold", "애터미체 Light", "애터미체 Medium", "도현체", "을지로체", "을지로10년후체", "한나체 Air", "한나체 Pro", "한나는11살체", "주아체", "기랑해랑체", "연성체", "바른바탕체 B", "바른바탕체 L", "바른바탕체 M", "바탕체", "넥슨 배찌체", "빙그레체", "빙그레 메로나체 Bold", "빙그레 메로나", "빙그레 싸만코체 Bold", "빙그레 싸만코체", "빙그레체Ⅱ",  "부산체", "카페24 당당해체", "카페24 단정해체", "카페24 동동체", "카페24 아네모네 에어체", "카페24 빛나는별체", "카페24 쑥쑥체", "카페24 써라운드체", "카페24 숑숑체", "창원단감아삭체 Bold", "쿠키런체 Black", "쿠키런체 Bold", "도스고딕", "도스이야기 굵은체", "도스명조", "도스필기", "도스샘물", "DX작가세상 M", "DX아리랑 B", "DX아우라",  "DX방탄고딕", "DX봄결 ExBold", "DX블루마린라운드 ExBold", "DX어린이그림", "DX동화나라 Bold", "DX퓨리티 Bold", "DX우등생 Bold", "DX한울 Bold", "DX헤드02 Bold", "DX설레임2 Medium", "DX설레임 Medium", "DX경필명조 Bold", "DX모던고딕 Bold", "DX모던고딕 RoundBold", "DX국민시대 Regular", "DX새신문명조 Bold", "DX프로방스 Bold", "DX르네상스 Bold", "DX단선고딕 Thin", "DX신문명조", "DX스피드 Medium", "DX우리강산 Bold", "동글 Bold", "동글 Light", "동글 Regular", "돋움체", "둥근모꼴", "EBS주시경B", "EBS주시경L", "EBS주시경M", "마초체", "엘리스디지털배움체 Regular", "고양덕양체 B", "고양덕양체 EB", "가비아 청연체", "가비아 마음결체","가나초콜릿체", "고도체 B", "고도체 M", "굴림체", "HY그래픽M", "HY견고딕", "HY중고딕", "HY헤드라인M", "HY견명조", "HY신명조", "HY얕은샘물M", "한컴 백제 B", "함초롬바탕체", "함초롬바탕체 B", "한컴 바겐세일 B", "한컴 바겐세일 M", "한솔체 B", "한솔체 M", "한컴 소망 B", "한컴 소망 M", "윤고딕 230", "윤고딕 240", "휴먼굵은팸체", "휴먼굵은샘체", "휴먼가는팸체", "휴먼가는샘체", "휴먼아미체", "휴먼고딕", "휴먼명조", "휴먼매직체", "휴먼옛체", "휴먼둥근헤드라인", "휴먼중간샘체", "HS새마을체 Regular", "HS겨울눈꽃체", "HS두꺼비체", "HS봄바람체 2.0", "HS새마을체", "HY 바다 L", "HY 바다 M", "HY 그래픽", "HY 강 B", "HY 강 M", "함렡체 Black", "함렡체 Bold", "함렡체 Light", "함렡체 Regular", "HanS 붐붐", "한글누리체", "한글누리체 R", "한겨레결체", "빛의계승자체 Bold", "빛의계승자체 Regular", "이롭게 바탕체", "정선아리랑체", "정선아리랑혼체", "정선아리랑뿌리체", "한글재민체", "제주고딕", "제주한라산", "제주명조", "KBIZ 한마음고딕 B", "KBIZ 한마음고딕 H", "KBIZ 한마음고딕 L", "KBIZ 한마음고딕 M", "KBIZ 한마음고딕 R", "KBIZ 한마음명조 B", "KBIZ 한마음명조 L", "KBIZ 한마음명조 M", "KBIZ 한마음명조 R", "KCC안중근체", "KCC임권택체", "KCC김훈체", "KCC은영체", "코트라 희망체", "코트라 도약체", "코트라 볼드체", "KoPub 바탕체 Bold", "KoPub 바탕체 Light", "KoPub 바탕체 Medium", "KoPub 돋움체 Bold", "KoPub 돋움체 Light", "KoPub 돋움체 Medium", "맑은 고딕", "메이플스토리 Bold", "메이플스토리 Light", "마포애민", "마포배낭여행", "마포다카포", "마포홍대프리덤", "마포마포나루", "미래로글꼴", "넥센타이어체 Bold", "넥센타이어체 Regular", "닉스곤체 B 2.0", "닉스곤체 M 2.0", "나눔손글씨 붓", "나눔고딕에코", "나눔고딕에코 Bold", "나눔고딕에코 ExtraBold", "나눔명조", "나눔명조에코", "나눔명조에코 Bold", "나눔명조에코 ExtraBold", "나눔손글씨 펜", "나눔스퀘어", "ON I고딕", "원스토어 모바일POP체", "포천 오성과한음체 Bold", "포천 오성과한음체 Regular", "푸른전남체 Bold", "푸른전남체 Medium", "푸른전남체", "평창평화체 Bold", "평창평화체 Light", "노회찬체", "상주다정다감체", "상주곶감체", "상주경천섬체", "산돌독수리체", "산돌이야기체", "세방고딕 Bold", "세방고딕", "SF망고빙수", "정묵바위체", "삼국지3글꼴", "서울남산체", "스포카 한 산스 Bold", "스포카 한 산스 Light", "스포카 한 산스 Regular", "스포카 한 산스 Thin", "순바탕 Bold", "순바탕 Light", "순바탕 Medium", "THE뉴스속보", "THE명품굴림B", "THE선인장", "THE정직", "태-조각TB", "태-으뜸B", "티몬체", "타이포 발레리나 B", "타이포 발레리나 M", "타이포 도담", "타이포 돈키왕자 M", "타이포 어울림 B", "타이포 어울림 L", "타이포 어울림 M", "타이포 홍익인간 M", "타이포 정조 M", "타이포 세종대왕 훈민", "타이포 달꽃", "타이포 씨고딕180", "타이포 씨명조180", "타이포 명탐정", "타이포 스톰 B", "어비 꿍디체", "어비 남지은체 Bold", "어비 남지은체", "어비 소윤체", "어비 선홍체 BOLD", "어비 선홍체", "어비 순수결정체 Bold", "어비 순수결정체", "어비 스윗체", "어비 나현체 Bold", "어비 나현체", "양재깨비체M", "Y이드스트릿체 B", "Y이드스트릿체 L", "영도체", "청소년체", "a타이틀고딕2", "a타이틀고딕3", "a타이틀고딕4",  "디자인하우스체", "디자인하우스체 Light", "영양군 음식디미방체", "설립체",  "가비아 봄바람체", "가비아 납작블럭체", "가비아 솔미체", "Headline", "로커스 상상고딕체", "넷마블체 B", "넷마블체 L", "넷마블체 M", "tvN 즐거운이야기 Bold", "tvN 즐거운이야기 Light", "tvN 즐거운이야기 Medium", "티웨이 항공체", "티웨이 날다체", "티웨이 하늘체", "양굵은구조고딕", "양평군체 B", "양평군체 L", "양평군체 M", "강한육군 Bold Vert", "강한육군 Bold", "강한육군 Medium Vert", "강한육군 Medium", "경기천년제목 Light", "고양일산 R", "국립박물관문화재단클래식B", "국립박물관문화재단클래식L", "국립박물관문화재단클래식M", "국립중앙도서관글자체", "나눔손글씨 가람연꽃", "나눔손글씨 갈맷글", "나눔손글씨 강부장님체", "나눔손글씨 고딕 아니고 고딩", "나눔손글씨 고려글꼴", "나눔손글씨 곰신체", "나눔손글씨 규리의 일기", "나눔손글씨 김유이체", "나눔손글씨 꽃내음", "나눔손글씨 끄트머리체", "나눔손글씨 다행체", "나눔손글씨 대광유리", "나눔손글씨 딸에게 엄마가", "나눔손글씨 반짝반짝 별", "나눔손글씨 세계적인 한글", "나눔손글씨 세아체", "나눔손글씨 세화체", "나눔손글씨 소방관의 기도", "나눔손글씨 시우 귀여워", "나눔손글씨 신혼부부", "나눔손글씨 아빠의 연애편지", "나눔손글씨 아줌마 자유", "나눔손글씨 엄마사랑", "나눔손글씨 와일드", "나눔손글씨 외할머니글씨", "나눔손글씨 유니 띵땅띵땅", "나눔손글씨 자부심지우", "나눔손글씨 잘하고 있어", "나눔손글씨 장미체", "나눔손글씨 점꼴체", "나눔손글씨 정은체", "나눔손글씨 중학생", "나눔손글씨 진주 박경아체", "나눔손글씨 철필글씨", "나눔손글씨 칼국수", "나눔손글씨 코코체", "나눔손글씨 한윤체", "나눔손글씨 행복한 도비", "나눔손글씨 혜준체", "나눔손글씨 희망누리", "나눔손글씨 흰꼬리수리", "동그라미재단B", "동그라미재단L", "동그라미재단M", "문화재돌봄체 Bold", "문화재돌봄체 Regular", "배스킨라빈스 B", "비트로 코어체", "비트로 프라이드체", "서울남산 장체 B", "서울남산 장체 BL", "서울남산 장체 EB", "서울남산 장체 L", "서울남산 장체 M", "서울한강 장체 B", "서울한강 장체 BL", "서울한강 장체 EB", "서울한강 장체 L", "서울한강 장체 M", "솔뫼 김대건 Light", "솔뫼 김대건 Medium", "솔인써니체", "양진체", "온글잎 경영체", "온글잎 만두몽키체", "온글잎 무궁체", "온글잎 민혜체", "온글잎 보현체", "온글잎 석영체", "온글잎 안될과학궤도체", "온글잎 안될과학약체", "온글잎 안될과학유니랩장체", "온글잎 윤우체", "온글잎 의연체", "온글잎 해솜체", "이순신돋움체B", "이순신돋움체L", "이순신돋움체M", "조선일보명조", "중나좋체 Light", "중나좋체 Medium", "한글틀고딕", "한수원_한돋음_B", "한돋음체 R", "한울림체 R", "해수체B", "해수체L", "해수체M", "행복고흥B", "행복고흥L", "행복고흥M", "헬스셋고딕Bold", "헬스셋고딕Light", "헬스셋조릿대Std"]
rendering_vector_sim_df.insert(0, 'fontname', linux_fontname)

#### 4-4. 입력이미지와 가장 유사한 폰트 도출
### 가장 유사한 폰트
most_sim_font_1st = rendering_vector_sim_df.iloc[[rendering_vector_df['similarity'].argmax()]].index[0]

### 추천 폰트 순위
font_search_rank = rendering_vector_sim_df[['fontname', 'similarity']].sort_values(by=['similarity'],ascending=False)
print(font_search_rank.iloc[0:10])

### 추천 폰트 순위 상위 10개
font_search_rank = font_search_rank['fontname'].iloc[0:10].tolist()
print(font_search_rank)

font_rendering_dir = input_path
if os.path.exists(font_rendering_dir):
  shutil.rmtree(font_rendering_dir, ignore_errors=True)