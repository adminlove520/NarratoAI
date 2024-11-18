import os
import ssl
import glob
import json
import time
import asyncio
import traceback
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import streamlit as st
from loguru import logger

from app.config import config
from app.models.schema import VideoClipParams
from app.services.script_service import ScriptGenerator
from app.utils import utils, check_script, vision_analyzer, video_processor, video_processor_v2
from webui.utils import file_utils


def get_batch_timestamps(batch_files, prev_batch_files=None):
    """
    获取一批文件的时间戳范围
    返回: (first_timestamp, last_timestamp, timestamp_range)
    
    文件名格式: keyframe_001253_000050.jpg
    其中 000050 表示 00:00:50 (50秒)
         000101 表示 00:01:01 (1分1秒)
         
    Args:
        batch_files: 当前批次的文件列表
        prev_batch_files: 上一个批次的文件列表，用于处理单张图片的情况
    """
    if not batch_files:
        logger.warning("Empty batch files")
        return "00:00", "00:00", "00:00-00:00"
        
    # 如果当前批次只有一张图片，且有上一个批次的文件，则使用上一批次的最后一张作为首帧
    if len(batch_files) == 1 and prev_batch_files and len(prev_batch_files) > 0:
        first_frame = os.path.basename(prev_batch_files[-1])
        last_frame = os.path.basename(batch_files[0])
        logger.debug(f"单张图片批次，使用上一批次最后一帧作为首帧: {first_frame}")
    else:
        # 提取首帧和尾帧的时间戳
        first_frame = os.path.basename(batch_files[0])
        last_frame = os.path.basename(batch_files[-1])
    
    # 从文件名中提取时间信息
    first_time = first_frame.split('_')[2].replace('.jpg', '')  # 000050
    last_time = last_frame.split('_')[2].replace('.jpg', '')    # 000101
    
    # 转换为分:秒格式
    def format_timestamp(time_str):
        # 时间格式为 MMSS，如 0050 表示 00:50, 0101 表示 01:01
        if len(time_str) < 4:
            logger.warning(f"Invalid timestamp format: {time_str}")
            return "00:00"
            
        minutes = int(time_str[-4:-2])  # 取后4位的前2位作为分钟
        seconds = int(time_str[-2:])    # 取后2位作为秒数
        
        # 处理进位
        if seconds >= 60:
            minutes += seconds // 60
            seconds = seconds % 60
            
        return f"{minutes:02d}:{seconds:02d}"
    
    first_timestamp = format_timestamp(first_time)
    last_timestamp = format_timestamp(last_time)
    timestamp_range = f"{first_timestamp}-{last_timestamp}"
    
    logger.debug(f"解析时间戳: {first_frame} -> {first_timestamp}, {last_frame} -> {last_timestamp}")
    return first_timestamp, last_timestamp, timestamp_range

def get_batch_files(keyframe_files, result, batch_size=5):
    """
    获取当前批次的图片文件
    """
    batch_start = result['batch_index'] * batch_size
    batch_end = min(batch_start + batch_size, len(keyframe_files))
    return keyframe_files[batch_start:batch_end]

def render_script_panel(tr):
    """渲染脚本配置面板"""
    with st.container(border=True):
        st.write(tr("Video Script Configuration"))
        params = VideoClipParams()

        # 渲染脚本文件选择
        render_script_file(tr, params)

        # 渲染视频文件选择
        render_video_file(tr, params)

        # 渲染视频主题和提示词
        render_video_details(tr)

        # 渲染脚本操作按钮
        render_script_buttons(tr, params)


def render_script_file(tr, params):
    """渲染脚本文件选择"""
    script_list = [(tr("None"), ""), (tr("Auto Generate"), "auto")]

    # 获取已有脚本文件
    suffix = "*.json"
    script_dir = utils.script_dir()
    files = glob.glob(os.path.join(script_dir, suffix))
    file_list = []

    for file in files:
        file_list.append({
            "name": os.path.basename(file),
            "file": file,
            "ctime": os.path.getctime(file)
        })

    file_list.sort(key=lambda x: x["ctime"], reverse=True)
    for file in file_list:
        display_name = file['file'].replace(config.root_dir, "")
        script_list.append((display_name, file['file']))

    # 找到保存的脚本文件在列表中的索引
    saved_script_path = st.session_state.get('video_clip_json_path', '')
    selected_index = 0
    for i, (_, path) in enumerate(script_list):
        if path == saved_script_path:
            selected_index = i
            break

    selected_script_index = st.selectbox(
        tr("Script Files"),
        index=selected_index,  # 使用找到的索引
        options=range(len(script_list)),
        format_func=lambda x: script_list[x][0]
    )

    script_path = script_list[selected_script_index][1]
    st.session_state['video_clip_json_path'] = script_path
    params.video_clip_json_path = script_path


def render_video_file(tr, params):
    """渲染视频文件选择"""
    video_list = [(tr("None"), ""), (tr("Upload Local Files"), "local")]

    # 获取已有视频文件
    for suffix in ["*.mp4", "*.mov", "*.avi", "*.mkv"]:
        video_files = glob.glob(os.path.join(utils.video_dir(), suffix))
        for file in video_files:
            display_name = file.replace(config.root_dir, "")
            video_list.append((display_name, file))

    selected_video_index = st.selectbox(
        tr("Video File"),
        index=0,
        options=range(len(video_list)),
        format_func=lambda x: video_list[x][0]
    )

    video_path = video_list[selected_video_index][1]
    st.session_state['video_origin_path'] = video_path
    params.video_origin_path = video_path

    if video_path == "local":
        uploaded_file = st.file_uploader(
            tr("Upload Local Files"),
            type=["mp4", "mov", "avi", "flv", "mkv"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            video_file_path = os.path.join(utils.video_dir(), uploaded_file.name)
            file_name, file_extension = os.path.splitext(uploaded_file.name)

            if os.path.exists(video_file_path):
                timestamp = time.strftime("%Y%m%d%H%M%S")
                file_name_with_timestamp = f"{file_name}_{timestamp}"
                video_file_path = os.path.join(utils.video_dir(), file_name_with_timestamp + file_extension)

            with open(video_file_path, "wb") as f:
                f.write(uploaded_file.read())
                st.success(tr("File Uploaded Successfully"))
                st.session_state['video_origin_path'] = video_file_path
                params.video_origin_path = video_file_path
                time.sleep(1)
                st.rerun()


def render_video_details(tr):
    """渲染视频主题和提示词"""
    video_theme = st.text_input(tr("Video Theme"))
    custom_prompt = st.text_area(
        tr("Generation Prompt"),
        value=st.session_state.get('video_plot', ''),
        help=tr("Custom prompt for LLM, leave empty to use default prompt"),
        height=180
    )
    st.session_state['video_theme'] = video_theme
    st.session_state['custom_prompt'] = custom_prompt
    return video_theme, custom_prompt


def render_script_buttons(tr, params):
    """渲染脚本操作按钮"""
    # 新增三个输入框，放在同一行
    input_cols = st.columns(3)
    
    with input_cols[0]:
        skip_seconds = st.number_input(
            "skip_seconds",
            min_value=0,
            value=st.session_state.get('skip_seconds', config.frames.get('skip_seconds', 0)),
            help=tr("Skip the first few seconds"),
            key="skip_seconds_input"
        )
        st.session_state['skip_seconds'] = skip_seconds
        
    with input_cols[1]:
        threshold = st.number_input(
            "threshold",
            min_value=0,
            value=st.session_state.get('threshold', config.frames.get('threshold', 30)),
            help=tr("Difference threshold"),
            key="threshold_input"
        )
        st.session_state['threshold'] = threshold
        
    with input_cols[2]:
        vision_batch_size = st.number_input(
            "vision_batch_size",
            min_value=1,
            max_value=20,
            value=st.session_state.get('vision_batch_size', config.frames.get('vision_batch_size', 5)),
            help=tr("Vision processing batch size"),
            key="vision_batch_size_input"
        )
        st.session_state['vision_batch_size'] = vision_batch_size

    # 生成/加载按钮
    script_path = st.session_state.get('video_clip_json_path', '')
    if script_path == "auto":
        button_name = tr("Generate Video Script")
    elif script_path:
        button_name = tr("Load Video Script")
    else:
        button_name = tr("Please Select Script File")

    if st.button(button_name, key="script_action", disabled=not script_path):
        if script_path == "auto":
            generate_script(tr, params)
        else:
            load_script(tr, script_path)

    # 视频脚本编辑区
    video_clip_json_details = st.text_area(
        tr("Video Script"),
        value=json.dumps(st.session_state.get('video_clip_json', []), indent=2, ensure_ascii=False),
        height=180
    )

    # 操作按钮行
    button_cols = st.columns(3)
    with button_cols[0]:
        if st.button(tr("Check Format"), key="check_format", use_container_width=True):
            check_script_format(tr, video_clip_json_details)

    with button_cols[1]:
        if st.button(tr("Save Script"), key="save_script", use_container_width=True):
            save_script(tr, video_clip_json_details)

    with button_cols[2]:
        script_valid = st.session_state.get('script_format_valid', False)
        if st.button(tr("Crop Video"), key="crop_video", disabled=not script_valid, use_container_width=True):
            crop_video(tr, params)


def check_script_format(tr, script_content):
    """检查脚本格式"""
    try:
        result = check_script.check_format(script_content)
        if result.get('success'):
            st.success(tr("Script format check passed"))
            st.session_state['script_format_valid'] = True
        else:
            st.error(f"{tr('Script format check failed')}: {result.get('message')}")
            st.session_state['script_format_valid'] = False
    except Exception as e:
        st.error(f"{tr('Script format check error')}: {str(e)}")
        st.session_state['script_format_valid'] = False


def load_script(tr, script_path):
    """加载脚本文件"""
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            script = f.read()
            script = utils.clean_model_output(script)
            st.session_state['video_clip_json'] = json.loads(script)
            st.success(tr("Script loaded successfully"))
            st.rerun()
    except Exception as e:
        st.error(f"{tr('Failed to load script')}: {str(e)}")


def generate_script(tr, params):
    """生成视频脚本"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress: float, message: str = ""):
        progress_bar.progress(progress)
        if message:
            status_text.text(f"{progress}% - {message}")
        else:
            status_text.text(f"进度: {progress}%")

    try:
        with st.spinner("正在生成脚本..."):
            if not params.video_origin_path:
                st.error("请先选择视频文件")
                return

            # 初始化脚本生成器
            script_generator = ScriptGenerator()
            
            # 获取配置参数
            vision_llm_provider = st.session_state.get('vision_llm_providers').lower()
            video_theme = st.session_state.get('video_theme', '')
            custom_prompt = st.session_state.get('custom_prompt', '')
            skip_seconds = st.session_state.get('skip_seconds', 0)
            threshold = st.session_state.get('threshold', 30)
            vision_batch_size = st.session_state.get('vision_batch_size', 5)

            # 创建异步事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 执行脚本生成
            script_result = loop.run_until_complete(
                script_generator.generate_script(
                    video_path=params.video_origin_path,
                    video_theme=video_theme,
                    custom_prompt=custom_prompt,
                    skip_seconds=skip_seconds,
                    threshold=threshold,
                    vision_batch_size=vision_batch_size,
                    vision_llm_provider=vision_llm_provider,
                    progress_callback=update_progress
                )
            )
            loop.close()

            # 保存生成的脚本
            st.session_state['video_clip_json'] = script_result
            update_progress(100, "脚本生成完成！")
            st.success("视频脚本生成成功！")

    except Exception as err:
        st.error(f"生成过程中发生错误: {str(err)}")
        logger.exception(f"生成脚本时发生错误\n{traceback.format_exc()}")
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()


def save_script(tr, video_clip_json_details):
    """保存视频脚本"""
    if not video_clip_json_details:
        st.error(tr("请输入视频脚本"))
        st.stop()

    with st.spinner(tr("Save Script")):
        script_dir = utils.script_dir()
        timestamp = time.strftime("%Y-%m%d-%H%M%S")
        save_path = os.path.join(script_dir, f"{timestamp}.json")

        try:
            data = json.loads(video_clip_json_details)
            with open(save_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
                st.session_state['video_clip_json'] = data
                st.session_state['video_clip_json_path'] = save_path

                # 更新配置
                config.app["video_clip_json_path"] = save_path

                # 显示成功消息
                st.success(tr("Script saved successfully"))

                # 强制重新加载页面更新选择框
                time.sleep(0.5)  # 给一点时间让用户看到成功消息
                st.rerun()

        except Exception as err:
            st.error(f"{tr('Failed to save script')}: {str(err)}")
            st.stop()


def crop_video(tr, params):
    """裁剪视频"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress):
        progress_bar.progress(progress)
        status_text.text(f"剪辑进度: {progress}%")

    try:
        utils.cut_video(params, update_progress)
        time.sleep(0.5)
        progress_bar.progress(100)
        status_text.text("剪辑完成！")
        st.success("视频剪辑成功完成！")
    except Exception as e:
        st.error(f"剪辑过程中发生错误: {str(e)}")
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()


def get_script_params():
    """获取脚本参数"""
    return {
        'video_language': st.session_state.get('video_language', ''),
        'video_clip_json_path': st.session_state.get('video_clip_json_path', ''),
        'video_origin_path': st.session_state.get('video_origin_path', ''),
        'video_name': st.session_state.get('video_name', ''),
        'video_plot': st.session_state.get('video_plot', '')
    }
