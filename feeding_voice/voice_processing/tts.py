from gtts import gTTS
from playsound import playsound
from ament_index_python.packages import get_package_share_directory
import os

package_path = get_package_share_directory("feeding_voice")
tts_file_list = ['not_recognized.mp3', 'menu_introducing.mp3']

NOT_RECOGNIZED = 0
MENU_INTRODUCING = 1

class TTS:
    def __init__(self, condition, target):
        if condition not in (NOT_RECOGNIZED, MENU_INTRODUCING):
            raise Exception(f'Invalid TTS Condition: {condition}')
        
        if condition == NOT_RECOGNIZED:
            if target == 'rice':
                subtext = '밥을'
            elif target == 'pororo' or target == 'loopy':
                subtext = '숟가락을'
            elif target == 'pikachu' or target == 'Bulbasaur':
                subtext = '포크를'
            elif target == 'Broccoli':
                subtext = '브로콜리를'
            elif target == 'Croissant':
                subtext = '크로와상을'
            elif target == 'apple':
                subtext = '사과를'
            text = f"카메라가 {subtext} 인식하지 못했습니다."
        elif condition == MENU_INTRODUCING:
            text = f"오늘의 메뉴는 {target} 입니다."
        else:
            pass
        
        self.tts = gTTS(text=text, lang='ko', slow=False)
        self.filepath = os.path.join(package_path, "resource", tts_file_list[condition])
        self.tts.save(self.filepath)
    
    def play(self):
        playsound(self.filepath)

# if __name__ == '__main__':
#     tts = TTS(MENU_INTRODUCING)
#     tts.play()