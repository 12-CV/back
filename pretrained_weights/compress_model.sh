#!/bin/bash

# 스크립트에 파일 이름 인수가 제공되었는지 확인
if [ "$#" -ne 1 ]; then
    echo "사용법: $0 {파일이름}"
    exit 1
fi

# 파일 이름 설정
FILENAME=$1

# 분할 압축 실행
7z a -v80m "./${FILENAME}.7z" "./${FILENAME}"

echo "${FILENAME}을(를) 분할 압축하였습니다."