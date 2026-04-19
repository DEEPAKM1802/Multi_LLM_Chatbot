#!/usr/bin/env python3
"""Executive Summary: This script demonstrates various Playwright capabilities by
performing tasks such as cross-browser testing, navigation, form interaction,
network interception, file upload/download, screenshots, multiple pages/tabs,
frames, authentication, cookies/localStorage, performance metrics, and parallel execution.
These examples correspond to criteria from the evaluation rubric above."""

from playwright.sync_api import sync_playwright, expect
import re  # for regex in assertions

# Configuration constants
TARGET_URL = "https://example.com"  # Replace with actual test site URL
UPLOAD_FILE = "test_upload.txt"
DOWNLOAD_PATH = "downloads"
HEADLESS = False  # Set to True for headless mode

def setup_browser():
    """Launch browser and return context and page objects."""
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=HEADLESS)
    context = browser.new_context()
    page = context.new_page()
    return playwright, browser, context, page

def teardown(playwright, browser, context):
    """Close context and stop Playwright."""
    try:
        context.close()
        browser.close()
        playwright.stop()
    except Exception as e:
        print(f"Teardown error: {e}")

def demo_cross_browser():
    # Cross-Browser Web Testing: launch Chromium, Firefox, WebKit
    with sync_playwright() as p:
        for browser_type in (p.chromium, p.firefox, p.webkit):
            browser = browser_type.launch(headless=HEADLESS)
            context = browser.new_context()
            page = context.new_page()
            page.goto(TARGET_URL)
            print(f"{browser_type.name} title: {page.title()}")
            browser.close()

def demo_navigation_and_interaction(page):
    # Navigation, selectors, forms, assertions
    page.goto(TARGET_URL)
    # Example: click a link (adjust selector per site)
    try:
        page.click('text=More information')
    except Exception as e:
        print(f"Click failed (might not exist on this site): {e}")
    # Check URL or perform assertion using expect
    try:
        expect(page).to_have_url(re.compile(r".*Example Domain.*"))
    except Exception as e:
        print(f"URL assertion failed: {e}")

def demo_waits_assertions(page):
    # Demonstrate waits and assertions
    page.goto(TARGET_URL)
    try:
        page.wait_for_selector('h1', timeout=5000)
        print("Heading is present")
    except Exception as e:
        print(f"Wait for selector failed: {e}")
    try:
        expect(page).to_have_title(re.compile(r"Example Domain"))
    except Exception as e:
        print(f"Title assertion failed: {e}")

def demo_network_interception(page):
    # Network interception: log requests/responses
    page.on("request", lambda request: print(f"-> Request: {request.method} {request.url}"))
    page.on("response", lambda response: print(f"<- Response: {response.status} {response.url}"))
    page.goto(TARGET_URL)
    # Example: wait for a specific API response
    try:
        with page.expect_response("**/example-path") as response_info:
            page.click('text=More information')
        response = response_info.value
        print(f"Captured response: status {response.status}")
    except Exception as e:
        print(f"Expected response not found: {e}")

def demo_file_upload(page):
    # File upload example
    page.goto(TARGET_URL + '/upload')
    try:
        page.set_input_files('input[type="file"]', UPLOAD_FILE)
        print('File upload set')
    except Exception as e:
        print(f"File upload failed: {e}")

def demo_file_download(context):
    # File download example
    page = context.new_page()
    page.goto(TARGET_URL + '/download')
    try:
        with page.expect_download() as download_info:
            page.click('text=Download')
        download = download_info.value
        print(f"Download saved to: {download.path()}")
    except Exception as e:
        print(f"Download failed: {e}")

def demo_screenshot(page):
    # Take a screenshot
    page.goto(TARGET_URL)
    try:
        page.screenshot(path='screenshot.png')
        print('Screenshot taken')
    except Exception as e:
        print(f"Screenshot error: {e}")

def demo_multi_page_tab(context):
    # Open multiple pages/tabs
    page1 = context.new_page()
    page2 = context.new_page()
    page1.goto(TARGET_URL + '/page1')
    page2.goto(TARGET_URL + '/page2')
    print(f"Titles: {page1.title()}, {page2.title()}")

def demo_frames(page):
    # Interact with an iframe
    page.goto(TARGET_URL + '/iframe_page')
    try:
        frame = page.frame(name='frameName')
        frame.click('text=Click')
        print('Clicked inside iframe')
    except Exception as e:
        print(f"Iframe interaction failed: {e}")

def demo_authentication():
    # Basic HTTP authentication【12†L95-L100】
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=HEADLESS)
    context = browser.new_context(http_credentials={"username": "user", "password": "pass"})
    page = context.new_page()
    page.goto(TARGET_URL + '/basic_auth')
    if "Congratulations" in page.content():
        print('Authentication succeeded')
    else:
        print('Auth may have failed')
    browser.close()
    playwright.stop()

def demo_cookies_and_local_storage(page):
    # Cookies and localStorage
    page.goto(TARGET_URL)
    page.context.add_cookies([{'name': 'test', 'value': '123', 'domain': 'example.com', 'path': '/'}])
    print('Cookies:', page.context.cookies())
    page.evaluate("localStorage.setItem('key', 'value')")
    print('LocalStorage key:', page.evaluate("localStorage.getItem('key')"))

def demo_performance_metrics(page):
    # Performance metrics via NavigationTiming API
    page.goto(TARGET_URL)
    timing = page.evaluate("JSON.stringify(window.performance.timing)")
    print('Navigation timing data:', timing)

def demo_parallel_execution():
    # Parallel-like demonstration using multiple contexts
    playwright = sync_playwright().start()
    contexts = [playwright.chromium.launch(headless=HEADLESS).new_context() for _ in range(2)]
    for i, ctx in enumerate(contexts):
        page = ctx.new_page()
        page.goto(TARGET_URL)
        print(f"Context {i} title: {page.title()}")
    for ctx in contexts:
        ctx.close()
    playwright.stop()

if __name__ == '__main__':
    playwright, browser, context, page = setup_browser()
    try:
        demo_cross_browser()
        demo_navigation_and_interaction(page)
        demo_waits_assertions(page)
        demo_network_interception(page)
        demo_file_upload(page)
        demo_file_download(context)
        demo_screenshot(page)
        demo_multi_page_tab(context)
        demo_frames(page)
        demo_authentication()
        demo_cookies_and_local_storage(page)
        demo_performance_metrics(page)
        demo_parallel_execution()
    except Exception as e:
        print(f'Error encountered: {e}')
    finally:
        teardown(playwright, browser, context)
