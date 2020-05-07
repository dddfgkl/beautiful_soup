# pytorch安装  

#### 1、下载安装anaconda3   
[anaconda版本以及其对应python版本](anaconda_version.png)   
[清华镜像:] https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/    
[官方历史版本:] https://repo.anaconda.com/archive/  

#### 2、添加清华镜像  
- conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/    
- conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge     
- conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/    

设置搜索的时候显示通道地址  
- conda config --set show_channel_urls yes  

#### 3、conda基本命令  
- conda env list 显示所有虚拟环境  
- conda create -n env_name python=3.6 新建虚拟环境并指定pythonv版本  
- source activate env_name    

#### 4、安装pytorch  

#### 5、安装tensorboardX   


