from app.storage import Storage
from app.ui import Ui

WINDOW_SIZE = (1900, 1000)

storage = Storage(1000, WINDOW_SIZE)
ui = Ui(WINDOW_SIZE, storage)
ui.start()
