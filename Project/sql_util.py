import pymysql
from sqlalchemy import create_engine
from mysql.connector import connect


# sqlalchemy - engine, connector
def get_sqlalchemy(db, is_both=False):
    """
    DESC:
    - SQLalchemy의 엔진 객체와 커넥트 객체 생성 및 반환
    
    @Params
    - db: database, 연결할 데이터베이스 지정
    
    - get_both: 엔진과 연결 객체 모두를 생성하고 반환할지 결정
                Default는 False로 엔진 객체만 생성하고 반환한다.
    """
    
    # DB 접속 엔진 객체 생성
    host = 'localhost'
    port = 3306
    user = 'root'
    password = 'MSQanfqud42!'

    engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4')
    # engine = create_engine('mysql+pymysql://root:@localhost:3306/ml_project?charset=utf8mb4')
    
    if is_both:
        conn = engine.connect()
        return engine, conn
    else:
        return engine
    
    def close():
        if is_both:
            connector.close()
        
        engine.dispose()
        
        # try:
        #     # 새로운 연결 시도
        #     with engine.connect() as connection:
        #         print("연결된 상태입니다.")
        # except OperationalError as e:
        #     print("연결이 해제되었습니다.:", e)


# conn = connect(
# 	host = "localhost",
# 	user = "root",
# 	password = "MSQanfqud42!",
# 	database = "ml_project"
# )
# cursor = conn.cursor(buffered=True)