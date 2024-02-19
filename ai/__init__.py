# from .ai import get_chat_reply, ChatError

# from .version import __version__

def init(image_model_name, smart_model_name, fast_model_name, api_key, db_connection_string, recordmanager_connection_string, max_tokens, image_max_tokens, temp=0.5):
    from .lib.lib_model import init as init_model
    init_model(image_model_name, smart_model_name, fast_model_name, api_key, db_connection_string, recordmanager_connection_string, max_tokens, image_max_tokens, temp)
    

# __all__ = [get_chat_reply, ChatError, init, __version__]
# __all__ = [__version__]