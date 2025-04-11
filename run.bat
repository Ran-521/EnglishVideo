@echo off
echo 视频编辑软件启动器
echo =====================

:: 检查Python是否已安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未安装Python，请先安装Python 3.7或更高版本
    echo 您可以从 https://www.python.org/downloads/ 下载安装
    pause
    exit /b 1
)

:: 检查依赖是否已安装
echo 正在检查依赖项...
echo 尝试安装moviepy...
pip install moviepy==1.0.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
echo 尝试安装PyQt5...
pip install PyQt5==5.15.9 -i https://pypi.tuna.tsinghua.edu.cn/simple
echo 尝试安装NumPy...
pip install --only-binary=:all: numpy==1.21.6 -i https://pypi.tuna.tsinghua.edu.cn/simple
echo 尝试安装Pillow...
pip install Pillow==10.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
echo 尝试安装opencv-python...
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 运行程序
echo 正在启动程序...
python video_editor.py

pause 