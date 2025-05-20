#!/usr/bin/env python
"""
불임 환자의 임신 성공 확률 예측 시스템

이 스크립트는 불임 환자의 임신 성공 확률을 예측하는 모델을 훈련하고 예측을 수행합니다.
"""

import os
import argparse
import subprocess

def main():
    """주요 기능을 실행하는 메인 함수"""
    parser = argparse.ArgumentParser(description="불임 환자의 임신 성공 확률 예측 시스템")
    
    parser.add_argument('--prepare_data', action='store_true', 
                        help='원본 데이터 파일을 data 디렉토리로 복사')
    parser.add_argument('--source_dir', type=str, default='./original_data', 
                        help='원본 데이터 디렉토리 (--prepare_data 옵션과 함께 사용)')
    
    parser.add_argument('--train', action='store_true', 
                        help='모델 훈련 및 예측 수행')
    parser.add_argument('--train_path', type=str, default='./data/train.csv', 
                        help='훈련 데이터 경로 (--train 옵션과 함께 사용)')
    parser.add_argument('--test_path', type=str, default='./data/test.csv', 
                        help='테스트 데이터 경로 (--train 옵션과 함께 사용)')
    parser.add_argument('--submission_path', type=str, default='./data/sample_submission.csv', 
                        help='샘플 제출 파일 경로 (--train 옵션과 함께 사용)')
    parser.add_argument('--output_dir', type=str, default='./output', 
                        help='결과물 저장 디렉토리 (--train 옵션과 함께 사용)')
    parser.add_argument('--models', nargs='+', default=['all'], 
                        choices=['rf', 'xgb', 'et', 'stacking', 'all'], 
                        help='학습할 모델 선택 (--train 옵션과 함께 사용)')
    
    parser.add_argument('--setup_github', action='store_true', 
                        help='GitHub 저장소 설정 및 푸시 준비')
    parser.add_argument('--repo_name', type=str, default='infertility-treatment-prediction', 
                        help='GitHub 저장소 이름 (--setup_github 옵션과 함께 사용)')
    parser.add_argument('--username', type=str, default=None, 
                        help='GitHub 사용자 이름 (--setup_github 옵션과 함께 사용)')
    
    args = parser.parse_args()
    
    # 데이터 준비
    if args.prepare_data:
        print("원본 데이터 파일을 data 디렉토리로 복사합니다...")
        from utils.copy_data import copy_data_files
        copy_data_files(args.source_dir, './data')
    
    # 모델 훈련 및 예측
    if args.train:
        print("모델 훈련 및 예측을 수행합니다...")
        from train import main as train_main
        train_args = argparse.Namespace(
            train_path=args.train_path,
            test_path=args.test_path,
            submission_path=args.submission_path,
            output_dir=args.output_dir,
            models=args.models
        )
        train_main(train_args)
    
    # GitHub 저장소 설정
    if args.setup_github:
        print("GitHub 저장소를 설정합니다...")
        from utils.setup_github import setup_github
        setup_github(args.repo_name, args.username)
    
    # 아무 옵션도 선택되지 않은 경우 도움말 출력
    if not (args.prepare_data or args.train or args.setup_github):
        parser.print_help()

if __name__ == '__main__':
    main() 