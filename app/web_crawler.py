import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

visited_urls = set()  # Set to keep track of visited URLs

def save_data_to_file(url, content, depth):
    # Create a directory for the depth level if it does not exist
    directory = f"level_new_{depth}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate a valid filename from the URL
    filename = os.path.join(directory, url.replace("https://", "").replace("/", "_") + ".txt")
    
    # Save the content to the file
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f"URL: {url}\n\n")
        file.write(content)

def crawl(url, depth, max_depth):
    if depth > max_depth:
        return
    
    if url in visited_urls:
        return  # Skip URLs that have already been visited
    
    visited_urls.add(url)  # Mark this URL as visited
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        content = response.text
        soup = BeautifulSoup(content, 'html.parser')
        
        # Save the content to a file
        save_data_to_file(url, soup.get_text(), depth)
        
        # Find all sub-links and recursively crawl them
        for link in soup.find_all('a', href=True):
            sub_link = urljoin(url, link['href'])
            crawl(sub_link, depth + 1, max_depth)
    except requests.RequestException as e:
        print(f"Failed to retrieve URL: {url} - {e}")

def main():
    parent_url = "https://docs.nvidia.com/cuda/"
    max_depth = 5
    crawl(parent_url, 0, max_depth)

if __name__ == "__main__":
    main()
