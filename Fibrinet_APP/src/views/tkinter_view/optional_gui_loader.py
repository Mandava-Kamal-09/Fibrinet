from utils.logger.logger import Logger

def get_gui_view_class():
    try:
        from src.views.tkinter_view.tkinter_view import TkinterView
        Logger.log("TkinterView successfully loaded.", Logger.LogPriority.DEBUG)
        return TkinterView
    except ImportError as e:
        Logger.log(f"Tkinter not available: {e}", Logger.LogPriority.WARNING)
        return None
    except Exception as e:
        Logger.log(f"Failed to load TkinterView: {e}", Logger.LogPriority.ERROR)
        return None