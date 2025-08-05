from gtts import gTTS
from playsound import playsound
from ament_index_python.packages import get_package_share_directory
import os

package_path = get_package_share_directory("feeding_voice")
tts_file_list = [
    'not_recognized.mp3',
    'menu_introducing.mp3',
    'deliver_food.mp3',
    'finished_eating.mp3',
    'voice_recognized.mp3',
]

NOT_RECOGNIZED      = 0
MENU_INTRODUCING    = 1
DELIVER_FOOD        = 2
FINISHED_EATING     = 3
VOICE_RECOGNIZED    = 4

subtext = {
    'rice': '밥을',
    'pororo': '숟가락을',
    'loopy': '숟가락을',
    'pikachu': '포크를',
    'Bulbasaur': '포크를',
    'Broccoli': '브로콜리를',
    'Croissant': '크로와상을',
    'apple': '사과를',
}

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class TTS:
    def save(self, condition, target):
        if condition not in (NOT_RECOGNIZED, MENU_INTRODUCING, DELIVER_FOOD, FINISHED_EATING, VOICE_RECOGNIZED):
            raise Exception(f'Invalid TTS Condition: {condition}')
        
        if condition == NOT_RECOGNIZED:
            text = f"카메라가 {subtext[target]} 인식하지 못했습니다."
        elif condition == MENU_INTRODUCING:
            text = f"오늘의 메뉴는 {target} 입니다."
        elif condition == DELIVER_FOOD:
            text = f"{subtext[target]} 가져다 드리겠습니다."
        elif condition == FINISHED_EATING:
            text = "식사가 마무리되었습니다. 잔반을 정리하고 식판을 수거 후 물과 물티슈를 제공해 드리겠습니다."
        else:
            text = '네, 말씀하세요.'
        
        self.tts = gTTS(text=text, lang='ko', slow=False)
        self.filepath = os.path.join(package_path, "resource", tts_file_list[condition])
        self.tts.save(self.filepath)
        return self
    
    def play(self):
        playsound(self.filepath)