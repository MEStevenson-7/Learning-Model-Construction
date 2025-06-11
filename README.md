# Project Gutenberg Book Scraper & Analyzer

This project is a Python-based tool designed to scrape, process, and analyze texts from [Project Gutenberg](https://www.gutenberg.org/), a digital library of free eBooks. It demonstrates techniques in web scraping, rate-limiting based on `robots.txt`, and natural language preprocessing.

## Features

- **Respectful scraping**: Implements crawl-delay timing based on `robots.txt` instructions.
- **Book downloading**: Retrieves full texts from Project Gutenberg URLs.
- **Text cleaning**: Removes headers, footers, and non-content elements.
- **Metadata extraction**: Uses regular expressions to parse book titles and author names.
- **Data handling**: Leverages `pandas` and `numpy` to process and analyze text statistics.

## Technologies Used

- Python 3
- pandas
- numpy
- requests
- re (Regular Expressions)
- pathlib
- time

