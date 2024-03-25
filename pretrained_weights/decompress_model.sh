#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "사용법: $0 {파일이름}"
    exit 1
fi

FILENAME=$1

7z x "./$FIRST_PART_FILE"

echo "압축 해제가 완료되었습니다."