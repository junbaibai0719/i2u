#!/usr/bin/env python
# coding: utf-8

# In[1]:


from japronto import Application
import json


# In[2]:


from jinja2 import FileSystemLoader,Environment
import os

def render(request,template,context):
    '''
    params:
        request->请求,type:request
        template->模板名称,type:str
        context->渲染数据,type:dict
    '''
    env = Environment(loader=FileSystemLoader(os.path.abspath('./templates')))    # 创建一个包加载器对象

    template = env.get_template(template)    # 获取一个模板文件
    text = template.render(context)   # 渲染
    return request.Response(text = text,mime_type='text/html')


# In[3]:


app = Application()
router = app.router


# In[4]:


async def home(request):
    return render(request,template='home.html',context={'title':'home'})
    
    
router.add_route('/',home,methods=['GET'])
router.add_route('/home',home,methods=['GET'])


# In[ ]:


app.run(port = 5252,debug=False,worker_num=2)


# In[ ]:




