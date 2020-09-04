import cv2
import numpy as np

# -- 정답 사진과 입력된 사진의 위치
ans_src = "5-rotated.png"
inp_src = "o5.jpg"

def findContours(img, s=2, k=1):
  contours, hierarchy = cv2.findContours(img, s, k)
  return contours[0]

def getSimilarityRate(img1, img2, k=14):
  ret = cv2.matchShapes(findContours(img1), findContours(img2), 1, 0)
  return round(100 - ret * 10, k)

# -- 이미지 불러오고 윤곽선 따기
inp = cv2.resize(cv2.imread(inp_src, cv2.IMREAD_GRAYSCALE), (300,300))
ans = cv2.resize(cv2.imread(ans_src, cv2.IMREAD_GRAYSCALE), (300,300), interpolation=cv2.INTER_LINEAR)
edge = cv2.Canny(cv2.blur(inp.copy(), (3,3)), 50, 150)

# -- 이미지 일치 비율 구하기
print("유사도 점수: {0}점".format(getSimilarityRate(ans, edge)))