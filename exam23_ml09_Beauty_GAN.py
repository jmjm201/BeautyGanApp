#!/usr/bin/env python
# coding: utf-8

# In[47]:


import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


# In[48]:


# dlib는 image 관련
# detector는 이미지의 얼굴을 찾아줌
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')


# In[49]:


img = dlib.load_rgb_image('./imgs/12.jpg')
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()


# In[50]:


img_result = img.copy()
dets = detector(img, 1)
# detector에 이미지 주면 얼굴 찾아줌
if len(dets) == 0:    # 사진에 얼굴이 없으면
    print('cannot find faces!')
else:                 # 사진에 얼굴이 있으면 사각형 그려줌
    fig, ax = plt.subplots(1, figsize=(16,10))
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor= 'r', facecolor = 'none')
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()


# In[51]:


# 얼굴에 점찍기 (눈코입 찾기)
fig, ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections()
for detection in dets:
    s = sp(img, detection)
    # sp가 점 5개 찍는 모델을 사용하기 때문에 지금은 점 5개 찍음
    # 근데 더 많이 찍는 모델도 존재 --> 그런 모델 사용하면 점 더 많이 찍음
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius= 3, edgecolor= 'r', facecolor= 'r')
        ax.add_patch(circle)
ax.imshow(img_result)
plt.show()

# In[52]:


# 얼굴 사진만 뽑아서 정렬
faces = dlib.get_face_chips(img, objs, size= 256, padding = 0.3) # padding 없으면 이미지들 사이의 여백이 없음. 줘야함
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20,16))
axes[0].imshow(img)
# 첫번째 꺼는 그냥 img (원본)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)
plt.show()

# In[53]:


# 함수화
def align_faces(img):
    dets = detector(img, 1) # 얼굴 찾기, dets에는 얼굴 정보가 저장
    objs = dlib.full_object_detections()    # 객체의 정보를 리턴하는데 지금은 비어있음, 이미지의 정보를 가지고 있는 객체임
                                            # 얼굴 뿐만 아니라, 신호등이다 뭐다, 여튼 사물의 위치를 저장
    for detection in dets:
        s = sp(img, detection)              # 여기서 s는 랜드마크(점) 5개의 정보
        objs.append(s)                      # objs에 점 5개의 위치정보가 들어감, for문 돌면서 모든 얼굴들의 점 정보가 들어감
    faces = dlib.get_face_chips(img, objs, size= 256, padding= 0.35)   # 얼굴 '영역'만 짤라서 이미지 생성
    return faces                           # 얼굴 이미지들만 리턴

test_img = dlib.load_rgb_image('./imgs/13.jpg') # 이미지를 불러옴
test_faces = align_faces(test_img)              # 함수에 적용
                                                # test_faces는 얼굴 이미지들
fig, axes = plt.subplots(1, len(test_faces)+1, figsize= (20, 16))      # 한 줄에 이미지의 얼굴 이미지들을 나열
axes[0].imshow(test_img)                        # 첫 번째 꺼는 원본
for i, face in enumerate(test_faces):
    axes[i+1].imshow(face)
plt.show()

# In[54]:


# 화장을 입힐 차례
# tensorflow는 keras와 사용 방법이 다름
# google에 beauty gan이라 검색하면 github에 코드 나옴. 그거 참조
sess = tf.Session()
sess.run(tf.global_variables_initializer())   # 모델 초기화
saver = tf.train.import_meta_graph('./models/model.meta')    # 모델 로드
saver.restore(sess, tf.train.latest_checkpoint('./models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')


# In[55]:


# 스케일링
def preprocess(img):
    return (img / 255. - 0.5) * 2

# 다시 원본으로 만들기
def deprocess(img):
    return (img + 1) / 2


# In[58]:


# 소스 이미지
img1 = dlib.load_rgb_image('./imgs/no_makeup/vSYYZ429.png')
img1_faces = align_faces(img1)

# 레퍼런스 이미지
img2 = dlib.load_rgb_image('./imgs/makeup/vFG56.png')
img2_faces = align_faces(img2)

fig, axes = plt.subplots(1, 2, figsize=(16, 10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()


# In[59]:


src_img = img1_faces[0] # 소스 이미지
ref_img = img2_faces[0] # 레퍼런스 이미지

# 스케일링
X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis= 0) # 입력 데이터로 쓰려고 차원 하나 늘려주기 (shape 맞춰주는 것)

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis= 0)

# output 출력
output = sess.run(Xs, feed_dict= {X:X_img, Y:Y_img})   # 딕셔너리로 소스이미지와 레퍼런스 이미지를 주면 output 생성해줌
output_img = deprocess(output[0])

# subpplot으로 결과 보기
fig, axes = plt.subplots(1, 3, figsize= (20, 10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()


# In[ ]:




