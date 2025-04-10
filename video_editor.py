import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
                             QWidget, QFileDialog, QLineEdit, QListWidget, QMessageBox, QProgressBar, 
                             QComboBox, QSpinBox, QCheckBox, QFrame, QGroupBox, QDoubleSpinBox, QTextEdit,
                             QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QDateTime
from PyQt5.QtGui import QPixmap, QImage, QIcon
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, ColorClip
import tempfile
import cv2
from PIL import Image
import time
import threading
import traceback

class VideoProcessor(QThread):
    """视频处理线程"""
    
    # 定义信号
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal(str)
    preview_frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_files, offset, 
                 header_img1_path=None, header_img1_x=0, header_img1_y=0, header_img1_scale=0.1,
                 header_img2_path=None, header_img2_x=0, header_img2_y=30, header_img2_scale=0.1,
                 preview_only=True, output_dir=None):
        """初始化处理器
        
        Args:
            video_files: 视频文件列表
            offset: 视频向下移动的像素数
            header_img1_path: 第一张头部图片路径
            header_img1_x: 第一张图片X坐标（右移距离）
            header_img1_y: 第一张图片Y坐标（下移距离）
            header_img1_scale: 第一张图片缩放比例
            header_img2_path: 第二张头部图片路径
            header_img2_x: 第二张图片X坐标（右移距离）
            header_img2_y: 第二张图片Y坐标（下移距离）
            header_img2_scale: 第二张图片缩放比例
            preview_only: 是否仅生成预览
            output_dir: 输出目录
        """
        super().__init__()
        
        self.video_files = video_files
        self.offset = offset
        
        # 头部图片1设置
        self.header_img1_path = header_img1_path
        self.header_img1_x = header_img1_x
        self.header_img1_y = header_img1_y
        self.header_img1_scale = header_img1_scale
        
        # 头部图片2设置
        self.header_img2_path = header_img2_path
        self.header_img2_x = header_img2_x
        self.header_img2_y = header_img2_y
        self.header_img2_scale = header_img2_scale

        self.preview_only = preview_only
        self.output_dir = output_dir
        
        self._stop_event = threading.Event()
    
    def run(self):
        """运行线程"""
        try:
            if self.preview_only:
                self.process_preview()
            else:
                self.process_videos()
        except Exception as e:
            self.error_occurred.emit(f"处理错误: {str(e)}")
            traceback.print_exc()
    
    def stop(self):
        """停止线程"""
        self._stop_event.set()
    
    def stopped(self):
        """检查线程是否已停止"""
        return self._stop_event.is_set()
    
    def process_preview(self):
        """处理预览帧"""
        try:
            if not self.video_files:
                self.error_occurred.emit("没有选择视频文件")
                return
            
            # 仅处理第一个视频文件
            video_path = self.video_files[0]
            
            # 使用moviepy打开视频
            try:
                video_clip = VideoFileClip(video_path)
                self.error_occurred.emit(f"已加载视频: {os.path.basename(video_path)}")
                self.error_occurred.emit(f"视频尺寸: {video_clip.size}")
            except Exception as e:
                self.error_occurred.emit(f"无法打开视频文件: {str(e)}")
                return
            
            # 获取视频尺寸
            video_width, video_height = video_clip.size
            
            # 获取第1秒的帧
            frame_time = min(1, video_clip.duration / 2)
            frame = video_clip.get_frame(frame_time)
            
            # 创建黑色背景
            background = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            
            # 将原始视频下移
            offset = min(self.offset, video_height - 10)  # 确保至少有一部分视频可见
            moved_frame = np.zeros_like(background)
            moved_frame[offset:, :] = frame[:video_height-offset, :]
            
            # 尝试加载并缩放第一张头部图片
            try:
                if self.header_img1_path and os.path.exists(self.header_img1_path):
                    header_img1 = ImageClip(self.header_img1_path)
                    # 计算缩放后的尺寸，保持宽高比
                    original_width, original_height = header_img1.size
                    scale_factor = original_width / video_width
                    target_height = int(original_height / scale_factor * self.header_img1_scale)
                    
                    # 根据缩放比例调整图片大小
                    header_img1 = header_img1.resize(width=video_width * self.header_img1_scale)
                    
                    # 创建图片遮罩
                    img1_array = header_img1.get_frame(0)
                    # 计算放置位置
                    x_pos = int(self.header_img1_x)
                    y_pos = int(self.header_img1_y)
                    
                    # 确保图片不会超出边界
                    img_h, img_w = img1_array.shape[:2]
                    if x_pos < 0: x_pos = 0
                    if y_pos < 0: y_pos = 0
                    if x_pos + img_w > video_width: img_w = video_width - x_pos
                    if y_pos + img_h > video_height: img_h = video_height - y_pos
                    
                    # 将图片叠加到帧上
                    if img_w > 0 and img_h > 0:
                        # 重新缩放图片确保尺寸正确
                        img1_resized = cv2.resize(img1_array, (img_w, img_h))
                        # 仅复制非透明区域
                        if img1_resized.shape[2] == 4:  # 如果有alpha通道
                            alpha = img1_resized[:, :, 3] / 255.0
                            for c in range(3):
                                moved_frame[y_pos:y_pos+img_h, x_pos:x_pos+img_w, c] = (
                                    moved_frame[y_pos:y_pos+img_h, x_pos:x_pos+img_w, c] * (1 - alpha) + 
                                    img1_resized[:, :, c] * alpha
                                )
                        else:
                            moved_frame[y_pos:y_pos+img_h, x_pos:x_pos+img_w] = img1_resized[:, :, :3]
                    
                    self.error_occurred.emit(f"加载图片1成功: {os.path.basename(self.header_img1_path)}")
                    self.error_occurred.emit(f"图片1尺寸: 原始={header_img1.size}, 缩放比例={self.header_img1_scale}, 调整后宽度={img_w}, 位置=({x_pos}, {y_pos})")
            except Exception as e:
                self.error_occurred.emit(f"处理图片1时出错: {str(e)}")
            
            # 尝试加载并缩放第二张头部图片
            try:
                if self.header_img2_path and os.path.exists(self.header_img2_path):
                    header_img2 = ImageClip(self.header_img2_path)
                    # 计算缩放后的尺寸，保持宽高比
                    original_width, original_height = header_img2.size
                    scale_factor = original_width / video_width
                    target_height = int(original_height / scale_factor * self.header_img2_scale)
                    
                    # 根据缩放比例调整图片大小
                    header_img2 = header_img2.resize(width=video_width * self.header_img2_scale)
                    
                    # 创建图片遮罩
                    img2_array = header_img2.get_frame(0)
                    # 计算放置位置
                    x_pos = int(self.header_img2_x)
                    y_pos = int(self.header_img2_y)
                    
                    # 确保图片不会超出边界
                    img_h, img_w = img2_array.shape[:2]
                    if x_pos < 0: x_pos = 0
                    if y_pos < 0: y_pos = 0
                    if x_pos + img_w > video_width: img_w = video_width - x_pos
                    if y_pos + img_h > video_height: img_h = video_height - y_pos
                    
                    # 将图片叠加到帧上
                    if img_w > 0 and img_h > 0:
                        # 重新缩放图片确保尺寸正确
                        img2_resized = cv2.resize(img2_array, (img_w, img_h))
                        # 仅复制非透明区域
                        if img2_resized.shape[2] == 4:  # 如果有alpha通道
                            alpha = img2_resized[:, :, 3] / 255.0
                            for c in range(3):
                                moved_frame[y_pos:y_pos+img_h, x_pos:x_pos+img_w, c] = (
                                    moved_frame[y_pos:y_pos+img_h, x_pos:x_pos+img_w, c] * (1 - alpha) + 
                                    img2_resized[:, :, c] * alpha
                                )
                        else:
                            moved_frame[y_pos:y_pos+img_h, x_pos:x_pos+img_w] = img2_resized[:, :, :3]
                    
                    self.error_occurred.emit(f"加载图片2成功: {os.path.basename(self.header_img2_path)}")
                    self.error_occurred.emit(f"图片2尺寸: 原始={header_img2.size}, 缩放比例={self.header_img2_scale}, 调整后宽度={img_w}, 位置=({x_pos}, {y_pos})")
            except Exception as e:
                self.error_occurred.emit(f"处理图片2时出错: {str(e)}")
            
            # 显示预览帧
            self.preview_frame_ready.emit(moved_frame)
            
            # 释放资源
            video_clip.close()
            
        except Exception as e:
            self.error_occurred.emit(f"生成预览时出错: {str(e)}")
            traceback.print_exc()
    
    def process_videos(self):
        """处理所有视频"""
        try:
            if not self.video_files:
                self.error_occurred.emit("没有选择视频文件")
                return
            
            if not self.output_dir:
                self.error_occurred.emit("未指定输出目录")
                return
            
            total_files = len(self.video_files)
            
            for idx, video_path in enumerate(self.video_files):
                if self.stopped():
                    self.error_occurred.emit("用户中断处理")
                    return
                
                try:
                    # 更新进度
                    progress = int((idx / total_files) * 100)
                    self.progress_updated.emit(progress)
                    
                    # 发送状态信息
                    self.error_occurred.emit(f"处理视频 {idx+1}/{total_files}: {os.path.basename(video_path)}")
                    
                    # 使用moviepy打开视频
                    video_clip = VideoFileClip(video_path)
                    
                    # 获取视频尺寸
                    video_width, video_height = video_clip.size
                    self.error_occurred.emit(f"视频尺寸: {video_clip.size}")
                    
                    # 创建背景
                    background = ColorClip(size=(video_width, video_height), color=(0, 0, 0))
                    background = background.set_duration(video_clip.duration)
                    
                    # 将原始视频下移
                    offset = min(self.offset, video_height - 10)  # 确保至少有一部分视频可见
                    moved_clip = video_clip.set_position((0, offset))
                    
                    # 创建合成视频剪辑列表
                    clips = [background, moved_clip]
                    
                    # 添加第一张头部图片
                    try:
                        if self.header_img1_path and os.path.exists(self.header_img1_path):
                            header_img1 = ImageClip(self.header_img1_path)
                            # 根据缩放比例调整图片大小
                            header_img1 = header_img1.resize(width=video_width * self.header_img1_scale)
                            # 设置位置和持续时间
                            header_img1 = header_img1.set_position((self.header_img1_x, self.header_img1_y))
                            header_img1 = header_img1.set_duration(video_clip.duration)
                            clips.append(header_img1)
                            
                            self.error_occurred.emit(f"添加图片1: 缩放比例={self.header_img1_scale}, 位置=({self.header_img1_x}, {self.header_img1_y})")
                    except Exception as e:
                        self.error_occurred.emit(f"添加图片1时出错: {str(e)}")
                    
                    # 添加第二张头部图片
                    try:
                        if self.header_img2_path and os.path.exists(self.header_img2_path):
                            header_img2 = ImageClip(self.header_img2_path)
                            # 根据缩放比例调整图片大小
                            header_img2 = header_img2.resize(width=video_width * self.header_img2_scale)
                            # 设置位置和持续时间
                            header_img2 = header_img2.set_position((self.header_img2_x, self.header_img2_y))
                            header_img2 = header_img2.set_duration(video_clip.duration)
                            clips.append(header_img2)
                            
                            self.error_occurred.emit(f"添加图片2: 缩放比例={self.header_img2_scale}, 位置=({self.header_img2_x}, {self.header_img2_y})")
                    except Exception as e:
                        self.error_occurred.emit(f"添加图片2时出错: {str(e)}")
                    
                    # 合成最终视频
                    final_clip = CompositeVideoClip(clips, size=video_clip.size)
                    
                    # 生成输出文件名
                    base_name = os.path.basename(video_path)
                    name, ext = os.path.splitext(base_name)
                    output_file = os.path.join(self.output_dir, f"{name}_processed{ext}")
                    
                    # 写入视频文件
                    self.error_occurred.emit(f"正在写入: {os.path.basename(output_file)}")
                    final_clip.write_videofile(
                        output_file,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile=os.path.join(self.output_dir, "temp-audio.m4a"),
                        remove_temp=True,
                        fps=video_clip.fps,
                        threads=4,
                        preset='medium',
                        logger=None  # 禁用moviepy的进度条
                    )
                    
                    # 释放资源
                    video_clip.close()
                    final_clip.close()
                    
                    self.error_occurred.emit(f"视频处理完成: {os.path.basename(output_file)}")
                
                except Exception as e:
                    self.error_occurred.emit(f"处理 {os.path.basename(video_path)} 时出错: {str(e)}")
                    traceback.print_exc()
            
            # 更新最终进度
            self.progress_updated.emit(100)
            self.processing_finished.emit(f"所有 {total_files} 个视频处理完成!")
        
        except Exception as e:
            self.error_occurred.emit(f"批处理视频时出错: {str(e)}")
            traceback.print_exc()


class VideoEditorApp(QMainWindow):
    """视频编辑器应用主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("英语视频编辑器")
        self.setMinimumSize(1200, 800)
        
        # 初始化属性
        self.video_files = []
        self.header_img1_path = "E:\\Onedrive\\KMS\\10-英语音频\\赵老师做好的\\01-图片\\剪映的标题back.png"
        self.header_img2_path = "E:\\Onedrive\\KMS\\10-英语音频\\赵老师做好的\\01-图片\\真的只是标题1.png"
        self.output_dir = None
        self.processor = None
        self.preview_image = None
        
        # 应用Mac风格
        self.apply_mac_style()
        
        # 初始化UI
        self.init_ui()
    
    def apply_mac_style(self):
        """应用Mac风格样式"""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f5f5f7;
                color: #333333;
                font-family: 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif;
            }
            
            QLabel {
                font-size: 14px;
                color: #333333;
            }
            
            QGroupBox {
                border: 1px solid #d1d1d6;
                border-radius: 8px;
                margin-top: 12px;
                font-weight: bold;
                background-color: #ffffff;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #666666;
            }
            
            QPushButton {
                background-color: #0071e3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                min-width: 80px;
            }
            
            QPushButton:hover {
                background-color: #0077ed;
            }
            
            QPushButton:pressed {
                background-color: #005bbf;
            }
            
            QPushButton:disabled {
                background-color: #cccccc;
                color: #999999;
            }
            
            QLineEdit, QSpinBox, QComboBox {
                border: 1px solid #d1d1d6;
                border-radius: 4px;
                padding: 6px;
                background-color: white;
                min-height: 24px;
            }
            
            QSpinBox::up-button, QSpinBox::down-button {
                border: none;
                width: 16px;
            }
            
            QProgressBar {
                border: 1px solid #d1d1d6;
                border-radius: 4px;
                text-align: center;
                background-color: #f0f0f0;
                min-height: 20px;
            }
            
            QProgressBar::chunk {
                background-color: #0071e3;
                border-radius: 3px;
            }
            
            QTextEdit {
                border: 1px solid #d1d1d6;
                border-radius: 4px;
                background-color: white;
                font-family: 'SF Mono', Menlo, Monaco, monospace;
                font-size: 12px;
            }
            
            QToolTip {
                border: 1px solid #d1d1d6;
                border-radius: 4px;
                background-color: #ffffffee;
                color: #333333;
                padding: 4px;
            }
        """)
    
    def init_ui(self):
        """初始化用户界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # 左侧布局（文件和设置）
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        # 右侧布局（预览和操作）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        
        # 调整两个面板的宽度比例
        main_layout.addWidget(left_panel, 40)  # 左侧面板占40%
        main_layout.addWidget(right_panel, 60)  # 右侧面板占60%
        
        # === 左侧布局 ===
        
        # 1. 视频文件选择组
        self.create_file_selection_group(left_layout)
        
        # 2. 视频设置组
        self.create_video_settings_group(left_layout)
        
        # 3. 图片设置组
        self.create_image_settings_group(left_layout)
        
        # 添加弹性空间
        left_layout.addStretch()
        
        # === 右侧布局 ===
        
        # 1. 预览组
        self.create_preview_group(right_layout)
        
        # 2. 操作组
        self.create_actions_group(right_layout)
        
        # 3. 日志组
        self.create_log_group(right_layout)
    
    def create_file_selection_group(self, parent_layout):
        """创建视频文件选择组"""
        group_box = QGroupBox("视频文件选择")
        layout = QVBoxLayout(group_box)
        layout.setSpacing(10)
        
        # 选择视频文件按钮
        self.select_videos_btn = QPushButton("选择视频文件")
        self.select_videos_btn.setIcon(QIcon.fromTheme("document-open"))
        self.select_videos_btn.clicked.connect(self.select_video_files)
        
        # 选择输出目录按钮
        self.select_output_btn = QPushButton("选择输出目录")
        self.select_output_btn.setIcon(QIcon.fromTheme("folder"))
        self.select_output_btn.clicked.connect(self.select_output_directory)
        
        # 文件列表
        self.files_list = QListWidget()
        self.files_list.setAlternatingRowColors(True)
        self.files_list.setMinimumHeight(150)
        
        # 布局
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.select_videos_btn)
        btn_layout.addWidget(self.select_output_btn)
        
        layout.addLayout(btn_layout)
        layout.addWidget(QLabel("已选择的视频文件:"))
        layout.addWidget(self.files_list)
        
        # 输出目录
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("输出目录:"))
        self.output_dir_label = QLabel("未选择")
        self.output_dir_label.setStyleSheet("font-style: italic; color: #666666;")
        out_layout.addWidget(self.output_dir_label, 1)
        
        layout.addLayout(out_layout)
        
        parent_layout.addWidget(group_box)
    
    def create_video_settings_group(self, parent_layout):
        """创建视频设置组"""
        group_box = QGroupBox("视频设置")
        layout = QVBoxLayout(group_box)
        layout.setSpacing(10)
        
        # 视频偏移设置
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("视频下移像素:"))
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(0, 500)
        self.offset_spin.setValue(100)
        self.offset_spin.setSingleStep(10)
        self.offset_spin.valueChanged.connect(self.update_preview)
        offset_layout.addWidget(self.offset_spin)
        
        layout.addLayout(offset_layout)
        
        parent_layout.addWidget(group_box)
    
    def create_image_settings_group(self, parent_layout):
        """创建图片设置组"""
        group_box = QGroupBox("图片设置")
        layout = QVBoxLayout(group_box)
        layout.setSpacing(10)
        
        # 图片1设置
        img1_group = QGroupBox("图片1设置")
        img1_layout = QGridLayout(img1_group)
        
        # 图片1路径
        img1_path_layout = QHBoxLayout()
        img1_path_layout.addWidget(QLabel("路径:"))
        self.img1_path_label = QLabel(os.path.basename(self.header_img1_path) if self.header_img1_path else "未选择")
        self.img1_path_label.setStyleSheet("font-style: italic; color: #666666;")
        img1_path_layout.addWidget(self.img1_path_label, 1)
        self.select_img1_btn = QPushButton("选择")
        self.select_img1_btn.setMaximumWidth(60)
        self.select_img1_btn.clicked.connect(lambda: self.select_image(1))
        img1_path_layout.addWidget(self.select_img1_btn)
        
        img1_layout.addLayout(img1_path_layout, 0, 0, 1, 2)
        
        # 图片1缩放
        img1_layout.addWidget(QLabel("缩放比例:"), 1, 0)
        self.img1_scale_spin = QDoubleSpinBox()
        self.img1_scale_spin.setRange(0.01, 2.0)
        self.img1_scale_spin.setValue(0.1)
        self.img1_scale_spin.setSingleStep(0.05)
        self.img1_scale_spin.valueChanged.connect(self.update_preview)
        img1_layout.addWidget(self.img1_scale_spin, 1, 1)
        
        # 图片1右移
        img1_layout.addWidget(QLabel("右移像素:"), 2, 0)
        self.img1_x_spin = QSpinBox()
        self.img1_x_spin.setRange(0, 1920)
        self.img1_x_spin.setValue(0)
        self.img1_x_spin.setSingleStep(10)
        self.img1_x_spin.valueChanged.connect(self.update_preview)
        img1_layout.addWidget(self.img1_x_spin, 2, 1)
        
        # 图片1下移
        img1_layout.addWidget(QLabel("下移像素:"), 3, 0)
        self.img1_y_spin = QSpinBox()
        self.img1_y_spin.setRange(0, 1080)
        self.img1_y_spin.setValue(0)
        self.img1_y_spin.setSingleStep(10)
        self.img1_y_spin.valueChanged.connect(self.update_preview)
        img1_layout.addWidget(self.img1_y_spin, 3, 1)
        
        # 图片2设置
        img2_group = QGroupBox("图片2设置")
        img2_layout = QGridLayout(img2_group)
        
        # 图片2路径
        img2_path_layout = QHBoxLayout()
        img2_path_layout.addWidget(QLabel("路径:"))
        self.img2_path_label = QLabel(os.path.basename(self.header_img2_path) if self.header_img2_path else "未选择")
        self.img2_path_label.setStyleSheet("font-style: italic; color: #666666;")
        img2_path_layout.addWidget(self.img2_path_label, 1)
        self.select_img2_btn = QPushButton("选择")
        self.select_img2_btn.setMaximumWidth(60)
        self.select_img2_btn.clicked.connect(lambda: self.select_image(2))
        img2_path_layout.addWidget(self.select_img2_btn)
        
        img2_layout.addLayout(img2_path_layout, 0, 0, 1, 2)
        
        # 图片2缩放
        img2_layout.addWidget(QLabel("缩放比例:"), 1, 0)
        self.img2_scale_spin = QDoubleSpinBox()
        self.img2_scale_spin.setRange(0.01, 2.0)
        self.img2_scale_spin.setValue(0.1)
        self.img2_scale_spin.setSingleStep(0.05)
        self.img2_scale_spin.valueChanged.connect(self.update_preview)
        img2_layout.addWidget(self.img2_scale_spin, 1, 1)
        
        # 图片2右移
        img2_layout.addWidget(QLabel("右移像素:"), 2, 0)
        self.img2_x_spin = QSpinBox()
        self.img2_x_spin.setRange(0, 1920)
        self.img2_x_spin.setValue(0)
        self.img2_x_spin.setSingleStep(10)
        self.img2_x_spin.valueChanged.connect(self.update_preview)
        img2_layout.addWidget(self.img2_x_spin, 2, 1)
        
        # 图片2下移
        img2_layout.addWidget(QLabel("下移像素:"), 3, 0)
        self.img2_y_spin = QSpinBox()
        self.img2_y_spin.setRange(0, 1080)
        self.img2_y_spin.setValue(30)
        self.img2_y_spin.setSingleStep(10)
        self.img2_y_spin.valueChanged.connect(self.update_preview)
        img2_layout.addWidget(self.img2_y_spin, 3, 1)
        
        # 添加到主布局
        layout.addWidget(img1_group)
        layout.addWidget(img2_group)
        
        parent_layout.addWidget(group_box)
    
    def create_preview_group(self, parent_layout):
        """创建预览组"""
        group_box = QGroupBox("预览")
        layout = QVBoxLayout(group_box)
        
        # 预览图像显示区域
        self.preview_label = QLabel("未生成预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(300)
        self.preview_label.setStyleSheet("background-color: #222222; color: #999999; border-radius: 4px;")
        
        # 生成预览按钮
        self.generate_preview_btn = QPushButton("生成预览")
        self.generate_preview_btn.clicked.connect(self.generate_preview)
        
        layout.addWidget(self.preview_label)
        layout.addWidget(self.generate_preview_btn, 0, Qt.AlignRight)
        
        parent_layout.addWidget(group_box)
    
    def create_actions_group(self, parent_layout):
        """创建操作组"""
        group_box = QGroupBox("操作")
        layout = QVBoxLayout(group_box)
        
        # 导出按钮
        self.export_btn = QPushButton("导出视频")
        self.export_btn.setIcon(QIcon.fromTheme("document-save"))
        self.export_btn.clicked.connect(self.export_videos)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        layout.addWidget(self.export_btn)
        layout.addWidget(self.progress_bar)
        
        parent_layout.addWidget(group_box)
    
    def create_log_group(self, parent_layout):
        """创建日志组"""
        group_box = QGroupBox("处理日志")
        layout = QVBoxLayout(group_box)
        
        # 日志显示区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        
        layout.addWidget(self.log_text)
        
        parent_layout.addWidget(group_box)
    
    def select_video_files(self):
        """选择视频文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            "选择视频文件", 
            "", 
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv)"
        )
        
        if files:
            self.video_files = files
            self.files_list.clear()
            for file in files:
                self.files_list.addItem(os.path.basename(file))
            self.log("已选择 {} 个视频文件".format(len(files)))
    
    def select_output_directory(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "选择输出目录", 
            ""
        )
        
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_label.setText(dir_path)
            self.log(f"已设置输出目录: {dir_path}")
    
    def select_image(self, img_number):
        """选择图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            f"选择图片{img_number}", 
            "", 
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        
        if file_path:
            if img_number == 1:
                self.header_img1_path = file_path
                self.img1_path_label.setText(os.path.basename(file_path))
            else:
                self.header_img2_path = file_path
                self.img2_path_label.setText(os.path.basename(file_path))
            
            self.log(f"已选择图片{img_number}: {file_path}")
            self.update_preview()
    
    def generate_preview(self):
        """生成预览"""
        if not self.video_files:
            QMessageBox.warning(self, "警告", "请先选择视频文件！")
            return
        
        # 检查图片是否存在
        if self.header_img1_path and not os.path.exists(self.header_img1_path):
            QMessageBox.warning(self, "警告", f"图片1不存在: {self.header_img1_path}")
            return
        
        if self.header_img2_path and not os.path.exists(self.header_img2_path):
            QMessageBox.warning(self, "警告", f"图片2不存在: {self.header_img2_path}")
            return
        
        # 清空预览
        self.preview_label.setText("正在生成预览...")
        self.preview_image = None
        
        # 创建并启动处理线程
        self.processor = VideoProcessor(
            self.video_files, 
            self.offset_spin.value(),
            self.header_img1_path, 
            self.img1_x_spin.value(), 
            self.img1_y_spin.value(),
            self.img1_scale_spin.value(),
            self.header_img2_path, 
            self.img2_x_spin.value(), 
            self.img2_y_spin.value(),
            self.img2_scale_spin.value(),
            preview_only=True
        )
        
        # 连接信号
        self.processor.preview_frame_ready.connect(self.update_preview_image)
        self.processor.error_occurred.connect(self.log)
        
        # 启动线程
        self.processor.start()
        
        self.log("正在生成预览...")
    
    def update_preview(self):
        """更新预览（用于参数变更时）"""
        if hasattr(self, 'processor') and self.processor and self.processor.isRunning():
            # 如果处理器正在运行，先停止它
            self.processor.stop()
            self.processor.wait()
        
        # 重新生成预览
        if self.video_files:
            self.generate_preview()
    
    def update_preview_image(self, frame):
        """更新预览图像"""
        # 将OpenCV格式的帧转换为QImage
        h, w, c = frame.shape
        bytes_per_line = 3 * w
        
        # 将BGR转换为RGB（OpenCV使用BGR，Qt使用RGB）
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 创建QImage
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 缩放图像以适应预览区域
        pixmap = QPixmap.fromImage(q_img)
        preview_size = self.preview_label.size()
        scaled_pixmap = pixmap.scaled(
            preview_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # 更新预览标签
        self.preview_label.setPixmap(scaled_pixmap)
        
        # 保存预览图像
        self.preview_image = frame
        
        self.log("预览已更新")
    
    def export_videos(self):
        """导出视频"""
        if not self.video_files:
            QMessageBox.warning(self, "警告", "请先选择视频文件！")
            return
        
        if not self.output_dir:
            QMessageBox.warning(self, "警告", "请先选择输出目录！")
            return
        
        # 检查图片是否存在
        if self.header_img1_path and not os.path.exists(self.header_img1_path):
            QMessageBox.warning(self, "警告", f"图片1不存在: {self.header_img1_path}")
            return
        
        if self.header_img2_path and not os.path.exists(self.header_img2_path):
            QMessageBox.warning(self, "警告", f"图片2不存在: {self.header_img2_path}")
            return
        
        # 设置按钮和进度条状态
        self.export_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # 创建并启动处理线程
        self.processor = VideoProcessor(
            self.video_files, 
            self.offset_spin.value(),
            self.header_img1_path, 
            self.img1_x_spin.value(), 
            self.img1_y_spin.value(),
            self.img1_scale_spin.value(),
            self.header_img2_path, 
            self.img2_x_spin.value(), 
            self.img2_y_spin.value(),
            self.img2_scale_spin.value(),
            preview_only=False,
            output_dir=self.output_dir
        )
        
        # 连接信号
        self.processor.progress_updated.connect(self.progress_bar.setValue)
        self.processor.processing_finished.connect(self.on_processing_finished)
        self.processor.error_occurred.connect(self.log)
        
        # 启动线程
        self.processor.start()
        
        self.log(f"开始处理 {len(self.video_files)} 个视频...")
    
    def on_processing_finished(self, message):
        """处理完成时的回调"""
        self.export_btn.setEnabled(True)
        self.log(message)
        QMessageBox.information(self, "完成", message)
    
    def log(self, message):
        """显示日志消息"""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        # 同时打印到控制台
        print(formatted_message)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if hasattr(self, 'processor') and self.processor and self.processor.isRunning():
            reply = QMessageBox.question(
                self,
                "确认退出",
                "处理正在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 停止处理线程
                self.processor.stop()
                self.processor.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoEditorApp()
    window.show()
    sys.exit(app.exec_()) 