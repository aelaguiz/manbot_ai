from .ai import get_chat_reply, ChatError
from .lib.lib_model import init as init_model

from .version import __version__

def init(firebase_config, model_name, api_key, db_connection_string, recordmanager_connection_string, temp=0.5):
    init_model(model_name, api_key, db_connection_string, recordmanager_connection_string, temp)
    

__all__ = [get_chat_reply, ChatError, init, __version__]