from pathlib import Path

APP_DIR = Path(__file__).parent.parent

MODELS_DIR = APP_DIR / 'models'

LOGS_DIR = APP_DIR / 'logs'

RESULT_SIR = APP_DIR / 'result'
CAMERA_RESULT_DIR = RESULT_SIR / 'camera'
VIDEO_RESULT_DIR = RESULT_SIR / 'video'
IMAGE_RESULT_DIR = RESULT_SIR / 'image'

RESOURCE_DIR = APP_DIR / 'resource'

def create_dirs():

    for dir in [
        MODELS_DIR, 
        LOGS_DIR, 
        RESULT_SIR, 
        CAMERA_RESULT_DIR, 
        VIDEO_RESULT_DIR, 
        IMAGE_RESULT_DIR,
        RESOURCE_DIR
    ]:
        
        dir.mkdir(
            parents = True, 
            exist_ok = True
        )

create_dirs()