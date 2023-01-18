#!/usr/bin/env python
# coding: utf-8

# ##### 1. 라이브러리 로드

import sys
import os
sys.path.append("../img2vec_pytorch")  # Adds higher directory to python modules path.
from img_to_vec import Img2Vec
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# ##### 2. 획요소 특징 벡터 파일 로드

bbichim_df = pd.read_csv('형태소별 특징벡터(.csv)/bbichim_feature_vector(all).csv').drop(['fontname'], axis=1)
buri_df = pd.read_csv('형태소별 특징벡터(.csv)/buri_feature_vector(all).csv').drop(['fontname'], axis=1)
kkeokim_df = pd.read_csv('형태소별 특징벡터(.csv)/kkeokim_feature_vector(all).csv').drop(['fontname'], axis=1)
kkokjijum_df = pd.read_csv('형태소별 특징벡터(.csv)/kkokjijum_feature_vector(all).csv').drop(['fontname'], axis=1)
sangtu_df = pd.read_csv('형태소별 특징벡터(.csv)/sangtu_feature_vector(all).csv').drop(['fontname'], axis=1)


# ##### 3. 입력이미지 형태소 특징벡터 추출

img2vec =Img2Vec(model="inception")

### 테스트 폰트 경로
input_path = './유사폰트추천 테스트용 형태소 이미지/'

list_pics = []
filenames = []
for file in os.listdir(input_path):
  filename = os.fsdecode(file)
  img = Image.open(os.path.join(input_path, filename))
  list_pics.append(img)
  filenames.append(filename)

vectors1 = img2vec.get_vec(list_pics)

pics = {}
for i, vec in enumerate(vectors1):
  pics[filenames[i]] = vec


# ### 3. PCA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def data_pca(df, n):
  scaler = StandardScaler()
  X_scaled=scaler.fit_transform(df)
  #n차원으로 축소
  pca=PCA(n_components=n)
  pca.fit(X_scaled)
  X_pca=pca.transform(X_scaled)
  df_pca =pd.DataFrame(X_pca)
  return df_pca


# ### 4. K-Means

from sklearn.cluster import KMeans
import numpy as np

def data_scaling(df_pca):
  X = np.array(df_pca)
  scaler = MinMaxScaler()
  data_scaled = scaler.fit_transform(X)
  return data_scaled

def kmeans(k, data_scaled, df_pca):
  model = KMeans(n_clusters=k, random_state=0)
  model.fit(data_scaled)
  df_pca['cluster'] = model.fit_predict(data_scaled)  
  return df_pca

def add_fontname(df_pca):
  fontname = ["10X10", "10X10Bold", "116angmuburi", "116watermelon", "12롯데마트드림Bold", "12롯데마트드림Light", "12롯데마트드림Medium", "12롯데마트행복Bold", "12롯데마트행복Light", "12롯데마트행복Medium", "20835804_도서관체_ttf", "20835814_SungDongGothicEB_ttf", "20835815_SungDongGothicB_ttf", "20950864_경기천년제목TTF_Bold", "20950866_경기천년제목TTF_Medium", "20950867_경기천년제목TTFV_Bold", "20950870_경기천년바탕TTF_Bold", "20950871_경기천년바탕TTF_Regular", "20950914_빛고을광주_Light_TTF", "20950916_빛고을광주_Medium_TTF", "20950918_빛고을광주_Bold_TTF", "20950938_Batang_TTF", "20956072_SuncheonR_TTF", "20956073_SuncheonB_TTF", "20957733_전주완판본 순B_TTF", "20957736_전주완판본 순L_TTF", "20957738_전주완판본 순R_TTF", "20957740_전주완판본 각B_TTF", "20957742_전주완판본 각L_TTF", "20957744_전주완판본 각R_TTF", "20958935_UTOIMAGE_유토이미지고딕_R_TTF", "20958949_UTOIMAGE_유토이미지별나라달님_TTF", "20958955_UTOIMAGE_유토이미지빨간우체통_TTF", "20958971_UTOIMAGE_플라워_TTF", "20958979_UTOIMAGE_유토이미지_TTF", "20958981_UTOIMAGE_유토이미지고딕_B_TTF", "20958983_UTOIMAGE_유토이미지고딕_L_TTF", "Arita-buriB", "Arita-buriL", "Arita-buriM", "Arita-buriSB", "Atomy-Bold", "Atomy-Light", "Atomy-Medium", "a타이틀고딕2", "a타이틀고딕3", "a타이틀고딕4", "BareunBatangB", "BareunBatangL", "BareunBatangM", "BatangChe", "Bazzi", "BinggraeChe", "BinggraeMelona-Bold", "BinggraeMelona", "BinggraeSamanco-Bold", "BinggraeSamanco", "BinggraeⅡ", "BMDOHYEON_ttf", "BMEuljiro10yearslater", "BMEULJIROTTF", "BMHANNAAir_ttf", "BMHANNAPro", "BMHANNA_11yrs_ttf", "BMJUA_ttf", "BMKIRANGHAERANG-TTF", "BMYEONSUNG_ttf", "BusanFont_Provisional", "Cafe24Dangdanghae", "Cafe24Danjunghae", "Cafe24Dongdong", "Cafe24Ohsquareair", "Cafe24Shiningstar", "Cafe24Ssukssuk", "Cafe24Syongsyong", "ChangwonDangamAsac-Bold_0712", "CookieRun Black", "CookieRun Bold", "designhouseBold", "designhouseLight", "dimibang", "Dongle-Bold", "Dongle-Light", "Dongle-Regular", "DOSGothic", "DOSIyagiBoldface", "DOSMyungjo", "DOSPilgi", "DOSSaemmul", "DotumChe", "DungGeunMo", "DXAriB-KSCpc-EUC-H", "DXAura-KSCpc-EUC-H", "DXAWM-KSCpc-EUC-H", "DXBangtango-KSCpc-EUC-H", "DXBomgExtraBold-KSCpc-EUC-H", "DXBrmRExtraBold-KSCpc-EUC-H", "DXChildsdrawing-KSCpc-EUC-H", "DXDnaraB-KSCpc-EUC-H", "DXFrtyB-KSCpc-EUC-H", "DXHead02B-KSCpc-EUC-H", "DXHfl2M-KSCpc-EUC-H", "DXHflM-KSCpc-EUC-H", "DXHSB-KSCpc-EUC-H", "DXHWrB-KSCpc-EUC-H", "DXKPMB-KSCpc-EUC-H", "DXMgoB-KSCpc-EUC-H", "DXMgoRB-KSCpc-EUC-H", "DXNPeriod-KSCpc-EUC-H", "DXNShnmB-KSCpc-EUC-H", "DXPRVB-KSCpc-EUC-H", "DXRESB-KSCpc-EUC-H", "DXSglinegoTh-KSCpc-EUC-H", "DXShnm-KSCpc-EUC-H", "DXSpdM-KSCpc-EUC-H", "DXWGSB-KSCpc-EUC-H", "EBS주시경B", "EBS주시경L", "EBS주시경M", "EF_MACHO(윈도우용_TTF)", "EliceDigitalBaeum_Regular", "establish Retrosans", "GabiaCheongyeon", "GabiaMaeumgyeol", "gabia_bombaram", "gabia_napjakBlock", "gabia_solmee", "Ghanachocolate", "GodoB", "GodoM", "GOYANGDEOGYANG B", "GOYANGDEOGYANG EB", "GulimChe", "H2GPRM", "H2GTRE", "H2GTRM", "H2HDRM", "H2MJRE", "H2MJSM", "H2SA1M", "Hahmlet-Black", "Hahmlet-Bold", "Hahmlet-Light", "Hahmlet-Regular", "HANBaekB", "HANBatang", "HANBatangB", "HangeulNuriB", "HangeulNuriR", "Hangyeoregyeolche", "HANSaleB", "HANSaleM", "HANSolB", "HANSolM", "HANSomaB", "HANSomaM", "HanS_BoomBoom", "HANYGO230", "HANYGO240", "headline", "HeirofLightBold", "HeirofLightRegular", "HMKBP", "HMKBS", "HMKLP", "HMKLS", "HMKMAMI", "HMKMG", "HMKMM", "HMKMMAG", "HMKMOLD", "HMKMRHD", "HMKMS", "HSSaemaul-Regular", "HS겨울눈꽃체", "HS두꺼비체", "HS봄바람체2.0", "HS새마을체", "HYBDAL", "HYBDAM", "HYGPRM", "HYKANB", "HYKANM", "HYMyeongJoE", "HYSinMyeongJo M", "IropkeBatangM", "Jaemin", "JejuGothic", "JejuHallasan", "JejuMyeongjo", "JSArirang", "JSArirangHON", "JSArirangPPURI", "KBIZ한마음고딕 B", "KBIZ한마음고딕 H", "KBIZ한마음고딕 L", "KBIZ한마음고딕 M", "KBIZ한마음고딕 R", "KBIZ한마음명조 B", "KBIZ한마음명조 L", "KBIZ한마음명조 M", "KBIZ한마음명조 R", "KCCAhnjunggeun(Windows용)", "KCCImkwontaek", "KCC김훈체(Windows용)", "KCC은영체(Windows용)", "KoPubWorld Batang Bold", "KoPubWorld Batang Light", "KoPubWorld Batang Medium", "KoPubWorld Dotum Bold", "KoPubWorld Dotum Light", "KoPubWorld Dotum Medium", "KOTRA HOPE_TTF", "KOTRA LEAP_TTF", "KOTRA_BOLD", "locus_sangsang", "MalgunGothic", "Maplestory Bold", "Maplestory Light", "MapoAgape", "MapoBackpacking", "MapoDacapo", "MapoHongdaeFreedom", "MapoMaponaru", "MiraeroNormal", "NanumBrush", "NanumGothicEco", "NanumGothicEcoBold", "NanumGothicEcoExtraBold", "NanumMyeongJo", "NanumMyeongjoEco", "NanumMyeongjoEcoBold", "NanumMyeongjoEcoExtraBold", "NanumPen", "NanumSquare", "netmarbleB", "netmarbleL", "netmarbleM", "NEXEN TIRE_Bold", "NEXEN TIRE_Regular", "NIXGONFONTS B 2.0", "NIXGONFONTS M 2.0", "ON I고딕", "ONE Mobile POP", "OSeongandHanEum-Bold", "OSeongandHanEum-Regular", "PureunJeonnam-Bold", "PureunJeonnam-Medium", "PureunJeonnam", "PyeongChangPeace-Bold", "PyeongChangPeace-Light", "ROEHOE-CHAN", "Sam3KRFont", "SANGJU Dajungdagam", "SANGJU Gotgam", "SANGJU Gyeongcheon Island", "SDDogSuRiBold", "SDLeeYaGi", "SEBANG Gothic Bold", "SEBANG Gothic", "SeoulNamsanvert", "SF망고빙수 TTF", "Spoqa Han Sans Bold", "Spoqa Han Sans Light", "Spoqa Han Sans Regular", "Spoqa Han Sans Thin", "SSRockRegular", "SunBatang-Bold", "SunBatang-Light", "SunBatang-Medium", "TaeFont TSTJktB", "TaeFont TSTOtmB", "THE_Nyuseusokbo", "THE명품굴림B", "THE선인장", "THE정직", "TmonMonsori", "tvN 즐거운이야기 Bold", "tvN 즐거운이야기 Light", "tvN 즐거운이야기 Medium", "tway_air", "tway_fly", "tway_sky", "Typo_BallerinaB", "Typo_BallerinaM", "Typo_Dodam", "Typo_DonkiPrinceM", "Typo_EoulrimB", "Typo_EoulrimL", "Typo_EoulrimM", "Typo_HongikinganM", "Typo_JeongJoM", "Typo_KingSejong_Hunmin", "Typo_MoonFlowerM", "Typo_SherlockM", "Typo_SSiGothic180", "Typo_SSiMyungJo180", "Typo_StormB", "UhBee GGoongD", "Uhbee NaHyun Bold", "Uhbee NaHyun", "UhBee Namjieun Bold", "UhBee Namjieun", "UhBee Soyun", "UhBee Sunhong BOLD", "UhBee Sunhong", "UhBee swit", "UhBee U JEONG Bold", "UhBee U JEONG", "yangfont02", "yangpyeong_B", "yangpyeong_L", "yangpyeong_M", "YdestreetB", "YdestreetL", "Yeongdo", "YGBI08", "Youth", "강한육군 Bold Vert", "강한육군 Bold", "강한육군 Medium Vert", "강한육군 Medium", "경기천년제목_Light", "고양일산 R", "국립박물관문화재단클래식B", "국립박물관문화재단클래식L", "국립박물관문화재단클래식M", "국립중앙도서관글자체", "나눔손글씨 가람연꽃", "나눔손글씨 갈맷글", "나눔손글씨 강부장님체", "나눔손글씨 고딕 아니고 고딩", "나눔손글씨 고려글꼴", "나눔손글씨 곰신체", "나눔손글씨 규리의 일기", "나눔손글씨 김유이체", "나눔손글씨 꽃내음", "나눔손글씨 끄트머리체", "나눔손글씨 다행체", "나눔손글씨 대광유리", "나눔손글씨 딸에게 엄마가", "나눔손글씨 반짝반짝 별", "나눔손글씨 세계적인 한글", "나눔손글씨 세아체", "나눔손글씨 세화체", "나눔손글씨 소방관의 기도", "나눔손글씨 시우 귀여워", "나눔손글씨 신혼부부", "나눔손글씨 아빠의 연애편지", "나눔손글씨 아줌마 자유", "나눔손글씨 엄마사랑", "나눔손글씨 와일드", "나눔손글씨 외할머니글씨", "나눔손글씨 유니 띵땅띵땅", "나눔손글씨 자부심지우", "나눔손글씨 잘하고 있어", "나눔손글씨 장미체", "나눔손글씨 점꼴체", "나눔손글씨 정은체", "나눔손글씨 중학생", "나눔손글씨 진주 박경아체", "나눔손글씨 철필글씨", "나눔손글씨 칼국수", "나눔손글씨 코코체", "나눔손글씨 한윤체", "나눔손글씨 행복한 도비", "나눔손글씨 혜준체", "나눔손글씨 희망누리", "나눔손글씨 흰꼬리수리", "동그라미재단B", "동그라미재단L", "동그라미재단M", "문화재돌봄체 Bold", "문화재돌봄체 Regular", "배스킨라빈스 B", "비트로 코어 TTF", "비트로 프라이드 TTF", "서울남산 장체B", "서울남산 장체BL", "서울남산 장체EB", "서울남산 장체L", "서울남산 장체M", "서울한강 장체B", "서울한강 장체BL", "서울한강 장체EB", "서울한강 장체L", "서울한강 장체M", "솔뫼 김대건 Light", "솔뫼 김대건 Medium", "솔인써니체", "양진체v0.9_ttf", "온글잎 경영체", "온글잎 만두몽키체", "온글잎 무궁체", "온글잎 민혜체", "온글잎 보현체", "온글잎 석영체", "온글잎 안될과학궤도체", "온글잎 안될과학약체", "온글잎 안될과학유니랩장체", "온글잎 윤우체", "온글잎 의연체", "온글잎 해솜체", "이순신돋움체B", "이순신돋움체L", "이순신돋움체M", "조선일보명조", "중나좋체 Light", "중나좋체 Medium", "한글틀고딕", "한수원_한돋음_B", "한수원_한돋음_R", "한수원_한울림_R", "해수체B", "해수체L", "해수체M", "행복고흥B", "행복고흥L", "행복고흥M", "헬스셋고딕Bold", "헬스셋고딕Light", "헬스셋조릿대Std"]
  fontname.append("input font")
  
  # 폰트 이름을 알때
  # fontname.append(filenames[0].replace('bbichim_', '').replace('.png', ''))
  
  df_pca.insert(0, 'fontname', fontname)
  return df_pca

def show_clustering_result(df_pca, k):
  clustering_result = []
  for cluster_group in range(k):
    fontlist_of_each_cluster = df_pca[df_pca['cluster'] == cluster_group]['fontname'].values.tolist()
    clustering_result.append(fontlist_of_each_cluster)

  # for cluster_group in range(k):
  #   print("group " + str(cluster_group), clustering_result[cluster_group])

  return clustering_result


# ### 형태소별 모델 세팅

bbichim_similarity_df = []
buri_similarity_df = []
kkeokim_similarity_df = []
kkokjijum_similarity_df = []
sangtu_similarity_df = []


# #### ① 삐침

# 삐침 클러스터링
if 'bbichim.png' in filenames:
  bbichim_df.loc[len(bbichim_df)] = pics['bbichim.png']

  bbichim_pca = data_pca(bbichim_df, 18)
  bbichim_scaled = data_scaling(bbichim_pca)
  bbichim_clustering = kmeans(8, bbichim_scaled, bbichim_pca)
  bbichim_clustering = add_fontname(bbichim_clustering)

  # 테스트 이미지 삐침의 클러스터
  bbichim_input_cluster = bbichim_clustering.iloc[[-1]]['cluster']

  # input 이미지와 같은 클러스터인 폰트 출력
  bbichim_input = bbichim_clustering.iloc[-1]
  bbichim_input_cluster = bbichim_input['cluster']
  print("bbichim_input_cluster : ", bbichim_input_cluster)
  same_cluster_input = bbichim_clustering[bbichim_clustering['cluster']==bbichim_input_cluster]
  bbichim_compare = same_cluster_input.iloc[:-1, 1:-1]

  bbichim_similarity = []
  for idx in range(bbichim_compare.shape[0]):
    bbichim_similarity.append(cosine_similarity(bbichim_clustering.iloc[[-1,]].iloc[:, 1:-1].values, bbichim_compare.iloc[[idx]].values)[0][0])

  bbichim_similarity_df = same_cluster_input.iloc[:-1, :-1]
  bbichim_similarity_df['similarity'] = bbichim_similarity
  bbichim_similarity_df.sort_values(by='similarity', ascending=False)


# #### ② 부리

if 'buri.png' in filenames:
  buri_df.loc[len(buri_df)] = pics['buri.png']

  buri_pca = data_pca(buri_df, 10)
  buri_scaled = data_scaling(buri_pca)
  buri_clustering = kmeans(7, buri_scaled, buri_pca)
  buri_clustering = add_fontname(buri_clustering)
  buri_clustering.iloc[[-1]]['cluster']

  buri_input = buri_clustering.iloc[-1]
  buri_input_cluster = buri_input['cluster']
  print("buri_input_cluster : ", buri_input_cluster)
  same_buri_cluster_input = buri_clustering[buri_clustering['cluster']==buri_input_cluster]
  buri_compare = same_buri_cluster_input.iloc[:-1, 1:-1]

  buri_similarity = []
  for idx in range(buri_compare.shape[0]):
    buri_similarity.append(cosine_similarity(buri_clustering.iloc[[-1,]].iloc[:, 1:-1].values, buri_compare.iloc[[idx]].values)[0][0])

  buri_similarity_df = same_buri_cluster_input.iloc[:-1, :-1]
  buri_similarity_df['similarity'] = buri_similarity
  buri_similarity_df.sort_values(by='similarity', ascending=False)


# #### ③ 꺾임

if 'kkeokim.png' in filenames:
  kkeokim_df.loc[len(kkeokim_df)] = pics['kkeokim.png']

  kkeokim_pca = data_pca(kkeokim_df, 20)
  kkeokim_scaled = data_scaling(kkeokim_pca)
  kkeokim_clustering = kmeans(11,  kkeokim_scaled, kkeokim_pca)
  kkeokim_clustering = add_fontname(kkeokim_clustering)
  kkeokim_clustering.iloc[[-1]]['cluster']

  kkeokim_input = kkeokim_clustering.iloc[-1]
  kkeokim_input_cluster = kkeokim_input['cluster']
  print("kkeokim_input_cluster : ", kkeokim_input_cluster)
  same_kkeokim_cluster_input = kkeokim_clustering[kkeokim_clustering['cluster']==kkeokim_input_cluster]
  kkeokim_compare = same_kkeokim_cluster_input.iloc[:-1, 1:-1]

  kkeokim_similarity = []
  for idx in range(kkeokim_compare.shape[0]):
    kkeokim_similarity.append(cosine_similarity(kkeokim_clustering.iloc[[-1,]].iloc[:, 1:-1].values, kkeokim_compare.iloc[[idx]].values)[0][0])

  kkeokim_similarity_df = same_kkeokim_cluster_input.iloc[:-1, :-1]
  kkeokim_similarity_df['similarity'] = kkeokim_similarity
  kkeokim_similarity_df.sort_values(by='similarity', ascending=False)


# #### ④ 꼭지점

if 'kkokjijum.png' in filenames:
  kkokjijum_df.loc[len(kkokjijum_df)] = pics['kkokjijum.png']

  kkokjijum_pca = data_pca(kkokjijum_df, 18)
  kkokjijum_scaled = data_scaling(kkokjijum_pca)
  kkokjijum_clustering = kmeans(8, kkokjijum_scaled, kkokjijum_pca)
  kkokjijum_clustering = add_fontname(kkokjijum_clustering)
  kkokjijum_clustering.iloc[[-1]]['cluster']

  kkokjijum_input = kkokjijum_clustering.iloc[-1]
  kkokjijum_input_cluster = kkokjijum_input['cluster']
  print("kkokjijum_input_cluster : ", kkokjijum_input_cluster)
  same_kkokjijum_cluster_input = kkokjijum_clustering[kkokjijum_clustering['cluster']==kkokjijum_input_cluster]
  kkokjijum_compare = same_kkokjijum_cluster_input.iloc[:-1, 1:-1]

  kkokjijum_similarity = []
  for idx in range(kkokjijum_compare.shape[0]):
    kkokjijum_similarity.append(cosine_similarity(kkokjijum_clustering.iloc[[-1,]].iloc[:, 1:-1].values, kkokjijum_compare.iloc[[idx]].values)[0][0])

  kkokjijum_similarity_df = same_kkokjijum_cluster_input.iloc[:-1, :-1]
  kkokjijum_similarity_df['similarity'] = kkokjijum_similarity
  kkokjijum_similarity_df.sort_values(by='similarity', ascending=False).iloc[:10]


# #### ⑤ 상투

if 'sangtu.png' in filenames:
  sangtu_df.loc[len(sangtu_df)] = pics['sangtu.png']

  sangtu_pca = data_pca(sangtu_df, 18)
  sangtu_scaled = data_scaling(sangtu_pca)
  sangtu_clustering = kmeans(5, sangtu_scaled, sangtu_pca)
  sangtu_clustering = add_fontname(sangtu_clustering)
  sangtu_clustering.iloc[[-1]]['cluster']

  sangtu_input = sangtu_clustering.iloc[-1]
  sangtu_input_cluster = sangtu_input['cluster']
  print("sangtu_input_cluster : ", sangtu_input_cluster)
  same_sangtu_cluster_input = sangtu_clustering[sangtu_clustering['cluster']==sangtu_input_cluster]
  sangtu_compare = same_sangtu_cluster_input.iloc[:-1, 1:-1]

  sangtu_similarity = []
  for idx in range(sangtu_compare.shape[0]):
    sangtu_similarity.append(cosine_similarity(sangtu_clustering.iloc[[-1,]].iloc[:, 1:-1].values, sangtu_compare.iloc[[idx]].values)[0][0])

  sangtu_similarity_df = same_sangtu_cluster_input.iloc[:-1, :-1]
  sangtu_similarity_df['similarity'] = sangtu_similarity
  sangtu_similarity_df.sort_values(by='similarity', ascending=False)


# ### 입력 형태소 이미지와 유사한 폰트 추천

# #### ① 비교 후보 폰트 

candidate_font = np.array([])
if len(bbichim_similarity_df) > 0:
  candidate_font = np.concatenate((candidate_font, bbichim_similarity_df['fontname'].values), axis=0)

if len(buri_similarity_df) > 0:
  candidate_font = np.concatenate((candidate_font, buri_similarity_df['fontname'].values), axis=0)

if len(kkeokim_similarity_df) > 0:
  candidate_font = np.concatenate((candidate_font, kkeokim_similarity_df['fontname'].values), axis=0)

if len(kkokjijum_similarity_df) > 0:
  candidate_font = np.concatenate((candidate_font, kkokjijum_similarity_df['fontname'].values), axis=0)

if len(sangtu_similarity_df) > 0:
  candidate_font = np.concatenate((candidate_font, sangtu_similarity_df['fontname'].values), axis=0)

candidate_font = list(set(candidate_font))
# print(len(candidate_font))


# #### ② 코사인유사도로 폰트 비교

all_df = pd.DataFrame()

if 'bbichim.png' in filenames:
  all_df = pd.concat([all_df, bbichim_df], axis=1)

if 'buri.png' in filenames:
  all_df = pd.concat([all_df, buri_df], axis=1)

if 'kkeokim.png' in filenames:
  all_df = pd.concat([all_df, kkeokim_df], axis=1)

if 'kkokjijum.png' in filenames:
  all_df = pd.concat([all_df, kkokjijum_df], axis=1)

if 'sangtu.png' in filenames:
  all_df = pd.concat([all_df, sangtu_df], axis=1)

all_df = add_fontname(all_df)
test_font_vec = all_df.iloc[[-1]].iloc[:, 1:]

font_recommendation_list = []
font_recommendation_sim = []

for i in range(all_df.shape[0]):
  if all_df.iloc[i]['fontname'] in candidate_font:
    font_recommendation_list.append(all_df.iloc[i]['fontname'])
    font_recommendation_sim.append(cosine_similarity(test_font_vec.values, all_df.iloc[[i]].iloc[:, 1:].values)[0][0])


# #### ③ 폰트 추천 결과

font_recommendation = pd.DataFrame(font_recommendation_list)
font_recommendation['similarity'] = font_recommendation_sim
font_recommendation.sort_values(by='similarity', ascending=False).iloc[:15]

# 폰트 추천 결과를 json으로 변환
font_recommendation_result = font_recommendation.sort_values(by='similarity', ascending=False).iloc[:15, 0].to_json(orient = 'values', force_ascii=False)
print(font_recommendation_result)

# 결과를 웹으로 전달(flask)
# 