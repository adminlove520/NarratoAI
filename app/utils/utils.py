import locale
import os
import traceback

import requests
import threading
from typing import Any
from loguru import logger
import streamlit as st
import json
from uuid import uuid4
import urllib3
from datetime import datetime, timedelta

from app.models import const
from app.utils import check_script
from app.services import material
from app.models.schema import VideoClipParams

urllib3.disable_warnings()


def get_response(status: int, data: Any = None, message: str = ""):
    obj = {
        "status": status,
    }
    if data:
        obj["data"] = data
    if message:
        obj["message"] = message
    return obj


def to_json(obj):
    try:
        # 定义一个辅助函数来处理不同类型的对象
        def serialize(o):
            # 如果对象是可序列化类型，直接返回
            if isinstance(o, (int, float, bool, str)) or o is None:
                return o
            # 如果对象是二进制数据，转换为base64编码的字符串
            elif isinstance(o, bytes):
                return "*** binary data ***"
            # 如果对象是字典，递归处理每个键值对
            elif isinstance(o, dict):
                return {k: serialize(v) for k, v in o.items()}
            # 如果对象是列表或元组，递归处理每个元素
            elif isinstance(o, (list, tuple)):
                return [serialize(item) for item in o]
            # 如果对象是自定义类型，尝试返回其__dict__属性
            elif hasattr(o, "__dict__"):
                return serialize(o.__dict__)
            # 其他情况返回None（或者可以选择抛出异常）
            else:
                return None

        # 使用serialize函数处理输入对象
        serialized_obj = serialize(obj)

        # 序列化处理后的对象为JSON符串
        return json.dumps(serialized_obj, ensure_ascii=False, indent=4)
    except Exception as e:
        return None


def get_uuid(remove_hyphen: bool = False):
    u = str(uuid4())
    if remove_hyphen:
        u = u.replace("-", "")
    return u


def root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def storage_dir(sub_dir: str = "", create: bool = False):
    d = os.path.join(root_dir(), "storage")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if create and not os.path.exists(d):
        os.makedirs(d)

    return d


def resource_dir(sub_dir: str = ""):
    d = os.path.join(root_dir(), "resource")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    return d


def task_dir(sub_dir: str = ""):
    d = os.path.join(storage_dir(), "tasks")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def font_dir(sub_dir: str = ""):
    d = resource_dir("fonts")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def song_dir(sub_dir: str = ""):
    d = resource_dir("songs")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def public_dir(sub_dir: str = ""):
    d = resource_dir(f"public")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def run_in_background(func, *args, **kwargs):
    def run():
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"run_in_background error: {e}")

    thread = threading.Thread(target=run)
    thread.start()
    return thread


def time_convert_seconds_to_hmsm(seconds) -> str:
    hours = int(seconds // 3600)
    seconds = seconds % 3600
    minutes = int(seconds // 60)
    milliseconds = int(seconds * 1000) % 1000
    seconds = int(seconds % 60)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(hours, minutes, seconds, milliseconds)


def text_to_srt(idx: int, msg: str, start_time: float, end_time: float) -> str:
    start_time = time_convert_seconds_to_hmsm(start_time)
    end_time = time_convert_seconds_to_hmsm(end_time)
    srt = """%d
%s --> %s
%s
        """ % (
        idx,
        start_time,
        end_time,
        msg,
    )
    return srt


def str_contains_punctuation(word):
    for p in const.PUNCTUATIONS:
        if p in word:
            return True
    return False


def split_string_by_punctuations(s):
    result = []
    txt = ""

    previous_char = ""
    next_char = ""
    for i in range(len(s)):
        char = s[i]
        if char == "\n":
            result.append(txt.strip())
            txt = ""
            continue

        if i > 0:
            previous_char = s[i - 1]
        if i < len(s) - 1:
            next_char = s[i + 1]

        if char == "." and previous_char.isdigit() and next_char.isdigit():
            # 取现1万，按2.5%收取手续费, 2.5 中的 . 不能作为换行标记
            txt += char
            continue

        if char not in const.PUNCTUATIONS:
            txt += char
        else:
            result.append(txt.strip())
            txt = ""
    result.append(txt.strip())
    # filter empty string
    result = list(filter(None, result))
    return result


def md5(text):
    import hashlib

    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_system_locale():
    try:
        loc = locale.getdefaultlocale()
        # zh_CN, zh_TW return zh
        # en_US, en_GB return en
        language_code = loc[0].split("_")[0]
        return language_code
    except Exception as e:
        return "en"


def load_locales(i18n_dir):
    _locales = {}
    for root, dirs, files in os.walk(i18n_dir):
        for file in files:
            if file.endswith(".json"):
                lang = file.split(".")[0]
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    _locales[lang] = json.loads(f.read())
    return _locales


def parse_extension(filename):
    return os.path.splitext(filename)[1].strip().lower().replace(".", "")


def script_dir(sub_dir: str = ""):
    d = resource_dir(f"scripts")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def video_dir(sub_dir: str = ""):
    d = resource_dir(f"videos")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def split_timestamp(timestamp):
    """
    拆分时间戳
    """
    start, end = timestamp.split('-')
    start_hour, start_minute = map(int, start.split(':'))
    end_hour, end_minute = map(int, end.split(':'))

    start_time = '00:{:02d}:{:02d}'.format(start_hour, start_minute)
    end_time = '00:{:02d}:{:02d}'.format(end_hour, end_minute)

    return start_time, end_time


def reduce_video_time(txt: str, duration: float = 0.21531):
    """
    按照字数缩减视频时长，一个字耗时约 0.21531 s,
    Returns:
    """
    # 返回结果四舍五入为整数
    duration = len(txt) * duration
    return int(duration)


def get_current_country():
    """
    判断当前网络IP地址所在的国家
    """
    try:
        # 使用ipapi.co的免费API获取IP地址信息
        response = requests.get('https://ipapi.co/json/')
        data = response.json()

        # 获取国家名称
        country = data.get('country_name')

        if country:
            logger.debug(f"当前网络IP地址位于：{country}")
            return country
        else:
            logger.debug("无法确定当前网络IP地址所在的国家")
            return None

    except requests.RequestException:
        logger.error("获取IP地址信息时发生错误，请检查网络连接")
        return None


def time_to_seconds(time_str: str) -> float:
    parts = time_str.split(':')
    if len(parts) == 2:
        m, s = map(float, parts)
        return m * 60 + s
    elif len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
    else:
        raise ValueError(f"Invalid time format: {time_str}")


def seconds_to_time(seconds: float) -> str:
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def calculate_total_duration(scenes):
    total_seconds = 0
    
    for scene in scenes:
        start, end = scene['timestamp'].split('-')
        start_time = datetime.strptime(start, '%M:%S')
        end_time = datetime.strptime(end, '%M:%S')
        
        duration = end_time - start_time
        total_seconds += duration.total_seconds()
    
    return total_seconds


def add_new_timestamps(scenes):
    """
    新增新视频的时间戳，并为"原生播放"的narration添加唯一标识符
    Args:
        scenes: 场景列表

    Returns:
        更新后的场景列表
    """
    current_time = timedelta()
    updated_scenes = []

    # 保存脚本前先检查脚本是否正确
    check_script.check_script(scenes, calculate_total_duration(scenes))

    for scene in scenes:
        new_scene = scene.copy()  # 创建场景的副本，以保留原始数据
        start, end = new_scene['timestamp'].split('-')
        start_time = datetime.strptime(start, '%M:%S')
        end_time = datetime.strptime(end, '%M:%S')
        duration = end_time - start_time

        new_start = current_time
        current_time += duration
        new_end = current_time

        # 将 timedelta 转换为分钟和秒
        new_start_str = f"{int(new_start.total_seconds() // 60):02d}:{int(new_start.total_seconds() % 60):02d}"
        new_end_str = f"{int(new_end.total_seconds() // 60):02d}:{int(new_end.total_seconds() % 60):02d}"

        new_scene['new_timestamp'] = f"{new_start_str}-{new_end_str}"

        # 为"原生播放"的narration添加唯一标识符
        if new_scene.get('narration') == "" or new_scene.get('narration') == None:
            unique_id = str(uuid4())[:8]  # 使用UUID的前8个字符作为唯一标识符
            new_scene['narration'] = f"原声播放_{unique_id}"

        updated_scenes.append(new_scene)

    return updated_scenes


def clean_model_output(output):
    # 移除可能的代码块标记
    output = output.strip('```json').strip('```')
    # 移除开头和结尾的空白字符
    output = output.strip()
    return output


def cut_video(params, progress_callback=None):
    try:
        task_id = str(uuid4())
        st.session_state['task_id'] = task_id

        if not st.session_state.get('video_clip_json'):
            raise ValueError("视频脚本不能为空")

        video_script_list = st.session_state['video_clip_json']
        time_list = [i['timestamp'] for i in video_script_list]
        
        total_clips = len(time_list)

        def clip_progress(current, total):
            progress = int((current / total) * 100)
            if progress_callback:
                progress_callback(progress)

        subclip_videos = material.clip_videos(
            task_id=task_id,
            timestamp_terms=time_list,
            origin_video=params.video_origin_path,
            progress_callback=clip_progress
        )

        if subclip_videos is None:
            raise ValueError("裁剪视频失败")

        st.session_state['subclip_videos'] = subclip_videos

        for i, video_script in enumerate(video_script_list):
            try:
                video_script['path'] = subclip_videos[video_script['timestamp']]
            except KeyError as err:
                logger.error(f"裁剪视频失败: {err}")
        logger.debug(f"裁剪视频成功，共生成 {total_clips} 个视频片段")
        logger.debug(f"视频片段路径: {subclip_videos}")
        return task_id, subclip_videos

    except Exception as e:
        logger.error(f"视频裁剪过程中发生错误: \n{traceback.format_exc()}")
        raise


def temp_dir(sub_dir: str = ""):
    """
    获取临时文件目录
    Args:
        sub_dir: 子目录名
    Returns:
        str: 临时文件目录路径
    """
    d = os.path.join(storage_dir(), "temp")
    if sub_dir:
        d = os.path.join(d, sub_dir)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def clear_keyframes_cache(video_path: str = None):
    """
    清理关键帧缓存
    Args:
        video_path: 视频文件路径，如果指定则只清理该视频的缓存
    """
    try:
        keyframes_dir = os.path.join(temp_dir(), "keyframes")
        if not os.path.exists(keyframes_dir):
            return
            
        if video_path:
            # 理指定视频的缓存
            video_hash = md5(video_path + str(os.path.getmtime(video_path)))
            video_keyframes_dir = os.path.join(keyframes_dir, video_hash)
            if os.path.exists(video_keyframes_dir):
                import shutil
                shutil.rmtree(video_keyframes_dir)
                logger.info(f"已清理视频关键帧缓存: {video_path}")
        else:
            # 清理所有缓存
            import shutil
            shutil.rmtree(keyframes_dir)
            logger.info("已清理所有关键帧缓存")
            
    except Exception as e:
        logger.error(f"清理关键帧缓存失败: {e}")


def init_resources():
    """初始化资源文件"""
    try:
        # 创建字体目录
        font_dir = os.path.join(root_dir(), "resource", "fonts")
        os.makedirs(font_dir, exist_ok=True)
        
        # 检查字体文件
        font_files = [
            ("SourceHanSansCN-Regular.otf", "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"),
            ("simhei.ttf", "C:/Windows/Fonts/simhei.ttf"),  # Windows 黑体
            ("simkai.ttf", "C:/Windows/Fonts/simkai.ttf"),  # Windows 楷体
            ("simsun.ttc", "C:/Windows/Fonts/simsun.ttc"),  # Windows 宋体
        ]
        
        # 优先使用系统字体
        system_font_found = False
        for font_name, source in font_files:
            if not source.startswith("http") and os.path.exists(source):
                target_path = os.path.join(font_dir, font_name)
                if not os.path.exists(target_path):
                    import shutil
                    shutil.copy2(source, target_path)
                    logger.info(f"已复制系统字体: {font_name}")
                system_font_found = True
                break
        
        # 如果没有找到系统字体，则下载思源黑体
        if not system_font_found:
            source_han_path = os.path.join(font_dir, "SourceHanSansCN-Regular.otf")
            if not os.path.exists(source_han_path):
                download_font(font_files[0][1], source_han_path)
                
    except Exception as e:
        logger.error(f"初始化资源文件失败: {e}")

def download_font(url: str, font_path: str):
    """下载字体文件"""
    try:
        logger.info(f"正在下载字体文件: {url}")
        import requests
        response = requests.get(url)
        response.raise_for_status()
        
        with open(font_path, 'wb') as f:
            f.write(response.content)
            
        logger.info(f"字体文件下载成功: {font_path}")
        
    except Exception as e:
        logger.error(f"下载字体文件失败: {e}")
        raise
