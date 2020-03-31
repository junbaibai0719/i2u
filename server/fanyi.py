#!/usr/bin/env python
# coding: utf-8

# In[1]:


from japronto import Application
from translator import initial
import json
import asyncio

from multiprocessing import Process,Queue,Semaphore,Pipe
import concurrent.futures


# In[3]:


def process_fanyi(q0,q1,name):
    from translator import initial
    t = initial()
    while True:
        q = q0.get()
        print(q)
        res = t.predict(q['q'])
        q['res'] = res
        print(name)
        q1.put(q)
    

q_en = Queue(1000)
q_zh = Queue(1000)
for i in range(2):
    process = Process(target=process_fanyi,args=(q_en,q_zh,'process%s'%i))
    process.daemon = True
    process.start()

app = Application()
router = app.router

async def en2zh(request):
#     print('receive')
    if request.method == 'POST':
        data = request.form
        print('receive data :',data)
        q = data
        q_en.put(q)
        print('put')
        while True:
            try:
                tmp = q_zh.get(False)
                if tmp['q'] == data['q']:
                    data = tmp
                    break
                else:
                    q_zh.put(tmp)
                    continue
            except :
                await asyncio.sleep(1e-9)
                continue
        res = data
        return request.Response(json = {'res':res})
    else:
        return request.Response(text='hello world!')
    

router.add_route('/en2zh',en2zh,methods=['POST','GET'])
app.run(port = 5251)


# In[ ]:




