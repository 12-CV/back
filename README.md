
## Usage
### installation

1. 터미널에서 아래 명령어 실행
```
apt-get install p7zip-full
```
2. 가상환경 활성화 후 아래 명령어 실행
```
pip install -r requirements_server.txt
```
3. 모델 압축 해제를 위해 'models'폴더 하위에 있는 README 방법대로 모델 압축 해제

4. 서버 실행
```
python fastapi_server.py --port {포트번호}

# example
python fastapi_server.py --port 30303
```
