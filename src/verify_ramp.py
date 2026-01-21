import time
from playwright.sync_api import sync_playwright

def verify_pit_ramp_integration():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Wait for Streamlit to start
        print("Waiting for Streamlit...")
        time.sleep(5)

        try:
            page.goto("http://localhost:8501", timeout=30000)

            # Wait for title
            page.wait_for_selector("text=Open Pit Design Generator (PoC)", timeout=10000)
            print("App loaded.")

            # Click "Generate Pit Design" button
            # Note: The button key is "btn_gen_pit" but in UI it renders as button with text "Generate Pit Design"
            # It's in the sidebar.

            print("Clicking Generate Pit Design...")
            # We need to target the button specifically.
            # Using get_by_role is safer.
            page.get_by_role("button", name="Generate Pit Design").click()

            # Wait for generation to complete
            # We can wait for "Generated X benches" text in the expander or just wait a bit.
            # Since generation is fast for this PoC, 5 seconds should be enough.
            # But better to wait for a success indicator or the plot to update.
            # The diagnostics expander shows "Generated X benches".

            print("Waiting for generation...")
            # Expand the Diagnostics expnader to see the text?
            # Streamlit expanders are tricky.
            # Let's just wait for the success/error message or check if the plot rendered.
            # If there is an error, it shows "Generation Error".

            time.sleep(5)

            if page.get_by_text("Generation Error").count() > 0:
                print("Error detected!")
                print(page.get_by_text("Generation Error").text_content())
            else:
                print("Generation likely successful.")

            # Take screenshot
            screenshot_path = "/home/jules/verification/pit_ramp_integration_v2.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"Screenshot saved to {screenshot_path}")

        except Exception as e:
            print(f"Verification failed: {e}")
            page.screenshot(path="/home/jules/verification/failure.png")

        finally:
            browser.close()

if __name__ == "__main__":
    verify_pit_ramp_integration()
