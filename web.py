
# import time
# import requests
# from lxml import html

# # Function to extract and format text from the webpage using lxml
# def extract_and_format_text(url):
#     # Send a GET request to the URL
#     response = requests.get(url)
    
#     # Check if the request was successful
#     if response.status_code == 200:
#         # Parse the HTML content using lxml
#         tree = html.fromstring(response.content)
        
#         # Remove unwanted elements (like scripts, styles, ads, etc.)
#         for element in tree.xpath('//script | //style | //aside | //header | //footer | //nav'):
#             element.getparent().remove(element)
        
#         # Extract all the text content from the parsed HTML
#         text_content = tree.text_content()
        
#         # Format the extracted text
#         # Strip leading/trailing whitespace and replace multiple spaces/newlines with a single space
#         formatted_text = ' '.join(text_content.split())
        
#         # Optional: Add some more formatting, like breaking paragraphs based on newlines
#         formatted_text = formatted_text.replace('. ', '.\n')  # Example: add newlines after each sentence
        
#         return formatted_text
#     else:
#         return f"Failed to retrieve webpage. Status code: {response.status_code}"

# # URL of the webpage you want to extract content from
# url = 'https://en.wikipedia.org/wiki/Museum'  # Replace with your URL

# # Call the function and print the formatted extracted content
# formatted_text = extract_and_format_text(url)
# print(formatted_text)
from playwright.async_api import async_playwright
from lxml import html
import asyncio

async def extract_and_format_text(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Navigate to the URL
        await page.goto(url)

        # Wait until the content has loaded (this waits for body content, adjust as needed)
        await page.wait_for_selector('body')

        # Get the full page content
        content = await page.content()

        # Parse the HTML content using lxml
        tree = html.fromstring(content)

        # Remove unwanted elements (like scripts, styles, etc.)
        for element in tree.xpath('//script | //style | //aside | //header | //footer | //nav | //form'):
            element.getparent().remove(element)

        # Extract and format the visible text content
        text_content = tree.text_content()
        formatted_text = ' '.join(text_content.split())  # Clean up extra spaces and newlines
        formatted_text = formatted_text.replace('. ', '.\n')  # Optionally format sentences with newlines

        await browser.close()
        return formatted_text

# URL of the page you want to extract content from
url = 'https://en.wikipedia.org/wiki/Museum'  # Replace with your URL

# Run the extraction
async def main():
    formatted_text = await extract_and_format_text(url)
    print(formatted_text)

asyncio.run(main())


