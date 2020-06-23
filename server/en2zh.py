#!/usr/bin/env python
# coding: utf-8

# In[1]:


from japronto import Application
import json
import asyncio
import sys
sys.path.append("../")
from multiprocessing import Process,Queue,Semaphore,Pipe,Manager
# import multiprocessing as mp
#import concurrent.futures
#from concurrent.futures import ProcessPoolExecutor,as_completed


# In[2]:


def process_fanyi(q0,q1,name):
    from translator.translator import Translator
    t = Translator(0,0,"../translator/checkpoints/13_100_light","../translator/vocab/13_100_light")
    while True:
        q = q0.get()
        res = t.predict(q['q'])
        q['res'] = res
#         print(q)
#         print('%s finish %s'%(name,q))
        q1.put(q)


# In[3]:


process_num = 1
q_en = Queue(1000)
q_zh = Queue(1000)
for i in range(process_num):
    process = Process(target=process_fanyi,args=(q_en,q_zh,'process%s'%i))
    process.daemon = True
    process.start()


# In[4]:


app = Application()
router = app.router


# In[5]:


zh_dict = {}
async def en2zh(request):
#     print('receive')
    if request.method == 'POST':
        data = request.json
#         print('receive data :',data)
        q = data
        q_en.put(q)
#         print('put')
        while True:
            try:
                tmp = zh_dict.get(data['q'])
                if  tmp!=None:
                    data = tmp
                    del zh_dict[data['q']]
                    break
                else:
                    tmp = q_zh.get(False)
                    #取数放到字典里
                    zh_dict[tmp['q']] = tmp
                    await asyncio.sleep(1e-12)
                    continue
            except :
                await asyncio.sleep(1e-12)
                continue
        res = data
#         print('return:',res)
        return request.Response(json = {'res':res})
    else:
        return request.Response(text='hello world!')

router.add_route('/en2zh',en2zh,methods=['POST','GET'])
router.add_route('/i2u/en2zh',en2zh,methods=['POST','GET'])
router.add_route('/',en2zh,methods=['POST','GET'])


# In[6]:


app.run(port = 5251,debug=True,worker_num=1)


# In[ ]:




