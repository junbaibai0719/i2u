#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append("../")
from ocr.ocr import *
from .translator1 import Translator
import aiohttp


# In[3]:


import requests,re,json
# import execjs
from PIL import ImageFont, ImageDraw, Image


# In[4]:


class Word():
    def __init__(self, loc, word):
        self.loc = [int(i) for i in loc]
        self.word = word
        self.height = self.loc[3] - self.loc[1]


class Line():
    def __init__(self, loc):
        self.loc = [int(i) for i in loc]
        self.words = []
        self.flag = 0
        self.width = self.loc[2] - self.loc[0]
        self.height = self.loc[3] - self.loc[1]
        self.line = ''

    def append_word(self, word):
        self.words.append(word)

    def set_flag(self, flag: bool = True):
        if flag:
            self.flag = 1
        else:
            self.flag = -1

    #求单词的高度均值，用以获取字符大小
    def get_word_height(self):
        hli = [i.height for i in self.words]
        length = len(hli) 
        if length>2:
            #去掉最大和最小的值求均值
            self.height = (sum(hli) - max(hli) - min(hli)) / (len(self.words) - 2) 
            self.height = int(self.height)
        else:
            self.height = sum(hli) / length
            self.height = int(self.height)
        return self.height

    def get_line(self):
        if self.line == '':
            if self.words[-1].word[-1] == '-':
                self.words[-1].word = self.words[-1].word.replace('-', '')
            self.line = ' '.join([i.word for i in self.words])
        return self.line


class Paragraph():
    def __init__(self, loc):
        self.loc = [int(i) for i in loc]
        self.lines = []
        self.sentences = []
        self.sentences_trans = []

    def append_line(self, line):
        self.lines.append(line)

    def generate_sentence(self):
        sentence = ''
        for i in self.lines:
            sentence += i.get_line() + ' '
        self.sentences = re.findall(r'([\s\S]*?[\.!?]|.+$)', sentence)

    def scut(self, scut, s):
        l = []
        p = 0
        for i in range(len(s)):
            if s[i] in scut:
                l.append(s[p:i + 1])
                p = i + 1
        l.append(s[p:])
        return l

    def append_trans(self, s):
        self.sentences_trans.append(s)


class Article():
    def __init__(self):
        self.loc = []
        self.lines = []
        self.pars = []

    def append_par(self, par):
        self.pars.append(par)


#     def generate_para(self):
#         #判断行是段落首行还是尾行
#         self.start_end(0)
#         para = Paragraph()
#         for i in self.lines:
#             para.append_line(i)
#             if i.flag == -1:
#                 para.generate_sentence()
#                 self.paragraphs.append(para)
#                 para = Paragraph()

#     #判断行是段落首行还是尾行
#     def start_end(self, index):
#         first = self.lines[index]
#         if index < len(self.lines) - 1:
#             second = self.lines[index + 1]
#             height = (first.height + second.height) / 2
#             if abs(first.loc[0] - second.loc[0]) < height:
#                 self.start_end(index + 1)
#                 if second.flag == 1:
#                     first.set_flag(False)
#             elif first.loc[0] > second.loc[0]:
#                 first.set_flag(True)
#                 self.start_end(index + 1)
#             elif first.loc[0] < second.loc[0]:
#                 if second.loc[0] - first.loc[0] < first.width:
#                     first.set_flag(False)
#                     second.set_flag(True)
#                     self.start_end(index + 1)
#                 elif second.loc[0] - first.loc[0] > first.width:
#                     self.start_end(index + 1)
#                     if second.flag == 1:
#                         first.set_flag(False)
#         elif index == 0:
#             first.set_flag(True)
#         else:
#             first.set_flag(False)


# In[5]:


translator = Translator()


# In[6]:


import hashlib
import time
def bdfanyi(sentence):
#     with open('js/w.js') as fp:
#         js = fp.read()
#         sign = execjs.compile(js).call("e",senstence)
    url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    appid = '20200220000386346'
    passwd = 'fHBHjUYq7fSoQORhFajk'
    salt = str(int(time.time()))
    sign = hashlib.md5()
    sign.update(str.encode(appid+sentence+salt+passwd))
    data = {
        'from': 'en',
        'to': 'zh',
        'q': sentence,
        'appid': appid,
        'salt': salt,
        'sign': sign.hexdigest(),
    }
    header = {
        'accept': '*/*',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'zh,zh-TW;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'cookie':
        'REALTIME_TRANS_SWITCH=1; FANYI_WORD_SWITCH=1; HISTORY_SWITCH=1; SOUND_SPD_SWITCH=1; SOUND_PREFER_SWITCH=1; PSTM=1570954090; BIDUPSID=3D13806106D01525A58F71B301708744; APPGUIDE_8_2_2=1; BAIDUID=E7C1058784DB6F8D05064DD77B2A2A44:FG=1; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; to_lang_often=%5B%7B%22value%22%3A%22en%22%2C%22text%22%3A%22%u82F1%u8BED%22%7D%2C%7B%22value%22%3A%22zh%22%2C%22text%22%3A%22%u4E2D%u6587%22%7D%5D; Hm_lvt_64ecd82404c51e03dc91cb9e8c025574=1581489800,1581574170,1581860840,1581911128; BDSFRCVID=HELsJeCCxG3eijQus6DEWhqc-dwMojCnS8uw3J; H_BDCLCKID_SF=tR32Wn7a5TrDHJTg5DTjhPrMy-orbMT-027OKKOH2RcGft8CQMoRMb_SqfOfW4RB0KogVP55thF0HPonHjLajjvB3J; BDUSS=Wstb0VlQzEzdDBLSmN1d2g3NzVnT1dqcFVSMEdidThBR0V2emhLVnNsUy13M0ZlRUFBQUFBJCQAAAAAAAAAAAEAAADqJpN~sK7By8jGsKHIxgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAL42Sl6-NkpeU; from_lang_often=%5B%7B%22value%22%3A%22zh%22%2C%22text%22%3A%22%u4E2D%u6587%22%7D%2C%7B%22value%22%3A%22spa%22%2C%22text%22%3A%22%u897F%u73ED%u7259%u8BED%22%7D%2C%7B%22value%22%3A%22en%22%2C%22text%22%3A%22%u82F1%u8BED%22%7D%5D; delPer=0; PSINO=7; BDRCVFR[feWj1Vr5u3D]=I67x6TjHwwYf0; Hm_lpvt_64ecd82404c51e03dc91cb9e8c025574=1582023896; yjs_js_security_passport=36390990abc50ef6e92f5c9339d97bca5ab82cae_1582023898_js; __yjsv5_shitong=1.0_7_f9c71a0745bf534d43ebc4c2bb67549f5336_300_1582023899355_36.157.79.90_f99d559b; H_PS_PSSID=1447_21097_30793_26350',
        'origin': 'https://fanyi.baidu.com',
        'referer': 'https://fanyi.baidu.com/?aldtype=16047',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.106 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest'
    }
    response = requests.post(url=url, data=data, headers=header)
    js = json.loads(response.content)
    errorcode = js.get('error_code',None)
    if errorcode == None:
        result = js['trans_result'][0]['dst']
    else :
        result = sentence
    return result


# In[7]:


import aiohttp
# def i2u(sentence):
#     with 


# In[58]:


# ti = time.time()
# html = image_to_pdf_or_hocr(img=img)
# print('html generated',time.time()-ti)
# ti = time.time()
# soup = BeautifulSoup(html,'lxml')
# with open('test.html','w',encoding='utf8')as fp:
#     fp.write(html.decode())
# print('soup generated',time.time()-ti)


# In[8]:


def generate_article(soup):
    article = Article()
    for p in soup.findAll('p'):
        title = p['title']
        para = Paragraph(re.findall('([0-9]+)', title)[:4])
        for i in p.findAll('span'):
            if i['class'] == ['ocr_line']:
                title = i['title']
                line = Line(loc=re.findall('([0-9]+)', title)[:4])
                for j in i.findAll('span'):
                    w = Word(loc=re.findall('([0-9]+)', j['title'])[:4],
                             word=j.get_text())
                    line.append_word(w)
                para.append_line(line)
        para.generate_sentence()
        article.append_par(para)
    for para in article.pars:
        for s in para.sentences:
            if s != ' ':
                para.append_trans(translate(s))
            else:
                para.append_trans(s)
    return article


# In[48]:


def generate_trans_img(img,method:str = 'local'):
    if method == 'local':
        translate = translator.translate
    ti = time.time()
    html = image_to_pdf_or_hocr(img=img)
    print('html generated',time.time()-ti)
    ti = time.time()
    soup = BeautifulSoup(html,'lxml')
    with open('test.html','w',encoding='utf8')as fp:
        fp.write(html.decode())
    print('soup generated',time.time()-ti)
    ti = time.time()
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    
    article = Article()
    for p in soup.findAll('p'):
        title = p['title']
        para = Paragraph(re.findall('([0-9]+)', title)[:4])
        for i in p.findAll('span'):
            if i['class'] == ['ocr_line']:
                title = i['title']
                line = Line(loc=re.findall('([0-9]+)', title)[:4])
                for j in i.findAll('span'):
                    w = Word(loc=re.findall('([0-9]+)', j['title'])[:4],
                             word=j.get_text())
                    line.append_word(w)
                para.append_line(line)
        para.generate_sentence()
        
        '''
            index_line:行索引
            word_num：行上已涂写单词数
            p：已写入行的句子里的单词数
        '''
        index_line = 0
        word_num = 0
        #获取段落的坐标
        l,t,r,b = para.loc
        draw.rectangle((l,t,r,b),fill=(255,255,255))
        
        for s in para.sentences:
            p=0
            ti = time.time()
            if s != ' ':
                sentences_trans = translate(s)
            else:
                continue
            print(s,'generate s',sentences_trans,'cost',time.time()-ti)
            while index_line<len(para.lines):
                if p>=len(sentences_trans):
                    break
                line = para.lines[index_line]
                font_size = line.get_word_height()
                font = ImageFont.truetype('font/arialuni.ttf',
                                  font_size)
                line_size = int(line.width/font_size)
                l,t,r,b = line.loc
                l += font_size*word_num
                if len(sentences_trans[p:])>=(line_size-word_num):
                    draw.text((l,t,r,b),  sentences_trans[p:p+line_size-word_num], font = font, fill = (0, 0, 0))
                    p = p+line_size-word_num
                    word_num = 0
                    index_line+=1
                else:
                    draw.text((l,t,r,b),  sentences_trans[p:], font = font, fill = (0, 0, 0))
                    word_num += len(sentences_trans[p:])
                    break
        article.append_par(para)
    return np.array(img),article


# In[59]:


# img0 = cv2.imread('test.png')
# img = generate_trans_img(img0.copy(),translate = translator.translate)


# In[60]:


# cv2.imshow('origin',img0)
# cv2.imshow('translated',img)
# cv2.waitKey(0)


# In[ ]:




