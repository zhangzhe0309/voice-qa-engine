"""已更名重定向模块：实际功能已在 audio_window_mixer.py 中完整定义。
保留此类以保持历史调用兼容。
"""

import warnings
from audio_window_mixer import AudioWindowMixer

class PhonemeAlignerInjector(AudioWindowMixer):
    def __init__(self, *args, **kwargs):
        warnings.warn("PhonemeAlignerInjector 已重构更名为 AudioWindowMixer", DeprecationWarning)
        super().__init__(*args, **kwargs)
