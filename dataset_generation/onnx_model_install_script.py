#!/usr/bin/env python3
"""
ONNX Model Downloader Script

Downloads ONNX models from URLs listed in a text file and saves them to a specified directory.

Usage:
    python onnx_downloader.py --input <text_file> --output <output_directory>
    python onnx_downloader.py -i <text_file> -o <output_directory>

Example:
    python onnx_downloader.py -i "dataset_generation/text files for links/link1.txt" -o "models_onnx"
"""

import argparse
import os
import requests
import sys
from pathlib import Path
from urllib.parse import urlparse
import time

def extract_filename_from_url(url):
    """
    Extract the filename from a URL, preferring the last part of the path.
    
    Args:
        url (str): The URL to extract filename from
        
    Returns:
        str: The extracted filename
    """
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    
    # If no filename found, create one from the URL
    if not filename or not filename.endswith('.onnx'):
        # Try to find .onnx in the path
        path_parts = parsed.path.split('/')
        for part in reversed(path_parts):
            if part.endswith('.onnx'):
                filename = part
                break
        else:
            # Fallback: use last part and add .onnx if needed
            filename = path_parts[-1] if path_parts[-1] else 'model'
            if not filename.endswith('.onnx'):
                filename += '.onnx'
    
    return filename

def download_file(url, output_path, timeout=300):
    """
    Download a file from URL to the specified path.
    
    Args:
        url (str): URL to download from
        output_path (str): Local path to save the file
        timeout (int): Timeout in seconds for the download
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Downloading: {url}")
        
        # Stream download for large files
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Get file size if available
        file_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Show progress for large files
                    if file_size > 0:
                        progress = (downloaded / file_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✓ Successfully downloaded: {os.path.basename(output_path)}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Failed to download {url}: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error downloading {url}: {e}")
        return False

def parse_text_file(file_path):
    """
    Parse a text file and extract valid URLs.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        list: List of valid URLs
    """
    urls = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Basic URL validation
                if line.startswith(('http://', 'https://')):
                    urls.append(line)
                else:
                    print(f"Warning: Line {line_num} doesn't appear to be a valid URL: {line}")
    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    
    return urls

def main():
    parser = argparse.ArgumentParser(
        description="Download ONNX models from URLs listed in a text file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python onnx_downloader.py -i "dataset_generation/text files for links/link1.txt" -o "models_onnx"
  python onnx_downloader.py --input link2.txt --output downloaded_models
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Path to text file containing ONNX model URLs')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory to save downloaded models')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Download timeout in seconds (default: 300)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip downloading if file already exists')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Parse URLs from text file
    urls = parse_text_file(args.input)
    if not urls:
        print("No valid URLs found in the input file.")
        sys.exit(1)
    
    print(f"Found {len(urls)} URLs to download")
    
    # Download each model
    successful = 0
    failed = 0
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Processing: {url}")
        
        # Generate output filename
        filename = extract_filename_from_url(url)
        output_path = output_dir / filename
        
        # Skip if file exists and skip_existing is True
        if args.skip_existing and output_path.exists():
            print(f"⚠ Skipping (already exists): {filename}")
            continue
        
        # Download the file
        if download_file(url, output_path, args.timeout):
            successful += 1
        else:
            failed += 1
            # Remove partial download if it exists
            if output_path.exists():
                try:
                    output_path.unlink()
                except:
                    pass
        
        # Small delay between downloads to be respectful
        if i < len(urls):
            time.sleep(1)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Download Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(urls)}")
    print(f"  Output directory: {output_dir.absolute()}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()