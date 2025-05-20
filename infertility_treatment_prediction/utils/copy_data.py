import os
import shutil
import argparse

def copy_data_files(source_dir, target_dir):
    """
    Copy necessary data files from source to target directory
    
    Args:
        source_dir: Source directory containing original data files
        target_dir: Target directory for copied data files
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Files to copy
    files_to_copy = [
        'train.csv',
        'test.csv',
        'sample_submission.csv',
        '데이터 명세.xlsx'
    ]
    
    # Check in the source directory
    source_files = os.listdir(source_dir)
    for file in files_to_copy:
        if file in source_files:
            source_path = os.path.join(source_dir, file)
            target_path = os.path.join(target_dir, file)
            print(f"Copying {file} from {source_path} to {target_path}")
            shutil.copy2(source_path, target_path)
            print(f"Successfully copied {file}")
        else:
            print(f"Warning: {file} not found in {source_dir}")

def main():
    """Main function to handle command line arguments and copy data files"""
    parser = argparse.ArgumentParser(description="Copy data files for infertility treatment prediction project")
    parser.add_argument('--source_dir', type=str, required=True, 
                        help='Source directory containing original data files')
    parser.add_argument('--target_dir', type=str, default='../data', 
                        help='Target directory for copied data files')
    
    args = parser.parse_args()
    copy_data_files(args.source_dir, args.target_dir)

if __name__ == '__main__':
    main() 