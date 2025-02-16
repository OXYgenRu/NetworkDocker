import datetime

from database import DB, Container
from usecase import UseCase

if __name__ == '__main__':
    db = DB()
    use_case = UseCase(db)
    # db.create_tables()
    # now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # new_container = Container(None, now_time, now_time, 0, 0, 0, True, "wakawaka", "eeee")
    # db.create_container(new_container)
    db.read_optimizer(100)
