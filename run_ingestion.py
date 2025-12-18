from app.ingestion import crawl_and_ingest
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    sitemap = os.getenv("BOOK_SITEMAP_URL")
    if not sitemap:
        raise ValueError("Set BOOK_SITEMAP_URL in .env")
    crawl_and_ingest(sitemap)