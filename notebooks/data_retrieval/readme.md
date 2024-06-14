# Scrap Redfin.com

This notebook aims to scrape property listing links from Redfin.com, download associated images, and remove any duplicate images. The notebook is divided into three main sections: scraping listing links, downloading images, and removing duplicates.

## Prerequisites
Before running the notebook, ensure you have the following prerequisites:

- Python 3.x
The following Python libraries:
- requests
- BeautifulSoup
- concurrent.futures
- hashlib
- 
You can install the required libraries using pip:

```bash
pip install requests beautifulsoup4
```
Notebook Sections
1. **Scraping Listing Links**
This section includes functions to fetch property listing links from Redfin.com. It uses the requests library to make HTTP requests and BeautifulSoup to parse the HTML content.

Function: `fetch_listing_links_from_page(url)`
Fetches and returns listing links from a given Redfin page URL.
Loop: Iterates through multiple pages to gather all listing links.

2. **Downloading Images**
In this section, images associated with the scraped property links are downloaded. Concurrent downloading is implemented to speed up the process.

Function: `download_image(image_url, save_path)`
Downloads an image from a URL and saves it to a specified path.

Function: `get_image_urls(property_url)`
Extracts image URLs from a property page.

Function: `download_images_for_properties(property_urls, save_dir)`
Downloads images for all given property URLs and saves them in the specified directory.

3. **Removing Duplicate Images**
This section handles the removal of duplicate images by comparing file hashes.

Function: `file_hash(filepath)`
Computes and returns the MD5 hash of a file.

Function: `remove_duplicates(directory)`
Removes duplicate images from the specified directory.

## Usage
- Scrape Listing Links
- Download Images
- Remove Duplicates
