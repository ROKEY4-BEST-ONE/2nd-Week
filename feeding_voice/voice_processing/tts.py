from gtts import gTTS
from playsound import playsound
from ament_index_python.packages import get_package_share_directory
import os

class TTS:
    def __init__(self, text):
        self.tts = gTTS(text=text, lang='ko', slow=False)
        if 
        self.tts.save(file_path)
        self.
    
    def play(self):
        playsound(file_path)
        p
# 텍스트 정의
text = ""

# gTTS 객체 생성
# lang='ko'는 한국어를 의미하며, 다양한 언어 코드를 사용할 수 있습니다.
tts = gTTS(text=text, lang='ko', slow=False)

# 음성 파일로 저장
file_path = "hello.mp3"
tts.save(file_path)

# 음성 파일 재생
print("음성 파일 재생 중...")
playsound(file_path)

# 파일 삭제 (선택 사항)
os.remove(file_path)
print("음성 파일 삭제 완료.")