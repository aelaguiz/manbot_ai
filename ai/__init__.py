from .ai import get_chat_reply, ChatError
from .lib.lib_model import init as init_model
from .lib.lib_object_storage import init as init_object_storage

from .version import __version__

def init(firebase_config, model_name, api_key, db_connection_string, temp=0.5):
    init_object_storage(firebase_config)
    init_model(model_name, api_key, db_connection_string, temp)
    

__all__ = [get_chat_reply, ChatError, init, __version__]