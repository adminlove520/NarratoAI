import streamlit as st
import os
from loguru import logger

def render_review_panel(tr):
    """渲染视频审查面板"""
    with st.expander(tr("Video Check"), expanded=False):
        try:
            video_list = st.session_state.get('video_clip_json', [])
            subclip_videos = st.session_state.get('subclip_videos', {})
        except KeyError:
            video_list = []
            subclip_videos = {}

        # 计算列数和行数
        num_videos = len(video_list)
        cols_per_row = 3
        rows = (num_videos + cols_per_row - 1) // cols_per_row  # 向上取整计算行数

        # 使用容器展示视频
        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col in range(cols_per_row):
                index = row * cols_per_row + col
                if index < num_videos:
                    with cols[col]:
                        render_video_item(tr, video_list, subclip_videos, index)

def render_video_item(tr, video_list, subclip_videos, index):
    """渲染单个视频项"""
    video_script = video_list[index]
    
    # 显示时间戳
    timestamp = video_script.get('timestamp', '')
    st.text_area(
        tr("Timestamp"),
        value=timestamp,
        height=70,
        disabled=True,
        key=f"timestamp_{index}"
    )
    
    # 显示视频播放器
    video_path = subclip_videos.get(timestamp)
    if video_path and os.path.exists(video_path):
        try:
            st.video(video_path)
        except Exception as e:
            logger.error(f"加载视频失败 {video_path}: {e}")
            st.error(f"无法加载视频: {os.path.basename(video_path)}")
    else:
        st.warning(tr("视频文件未找到"))
    
    # 显示画面描述
    st.text_area(
        tr("Picture Description"),
        value=video_script.get('picture', ''),
        height=150,
        disabled=True,
        key=f"picture_{index}"
    )
    
    # 显示旁白文本
    narration = st.text_area(
        tr("Narration"),
        value=video_script.get('narration', ''),
        height=150,
        key=f"narration_{index}"
    )
    # 保存修改后的旁白文本
    if narration != video_script.get('narration', ''):
        video_script['narration'] = narration
        st.session_state['video_clip_json'] = video_list
    
    # 修改 selectbox 的实现
    ost_options = [tr('keep_original_sound'), tr('mute')]
    default_index = 0  # 设置默认值为0
    
    ost = st.selectbox(
        tr('sound_settings'),
        options=ost_options,
        index=default_index,  # 使用安全的默认索引
        key=f"ost_{index}"
    )
    
    # 将选择转换为布尔值
    keep_original_sound = (ost == tr('keep_original_sound'))
    
    # 保存修改后的剪辑模式
    if keep_original_sound != video_script.get('keep_original_sound', True):
        video_script['keep_original_sound'] = keep_original_sound
        st.session_state['video_clip_json'] = video_list