from typing import List

# ANCHOR Imports
from selenium import webdriver
from datetime import datetime, date
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import logging
import pandas as pd
import time
from selenium.webdriver.common.action_chains import ActionChains


def get_reviews(url: str) -> pd.DataFrame:  # DATA

    # Open browser
    driver = webdriver.Chrome()

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a custom log format
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create a console handler and set the formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    # Open de pagina

    # LINK - Path 1 (Pre-defined Classes):
    # driver.get("https://www.coolblue.be/nl/product/935188/apple-iphone-15-128gb-zwart.html")
    driver.get(
        url
    )
    # driver.get("https://www.coolblue.be/nl/product/865894/apple-12w-usb-oplader.html")

    # LINK - Path 2 (Auto generated Classes):
    # driver.get("https://www.coolblue.be/nl/product/941558/bluebuilt-power-delivery-oplader-met-usb-c-poort-20w-wit.html")
    # driver.get("https://www.coolblue.be/nl/product/935450/wacom-bamboo-ink.html")
    driver.maximize_window()

    # Wacht tot de cookies geladen zijn

    time.sleep(2)

    try:
        # Zoek de cookie knop a.d.h.v. class naam
        accept_button = driver.find_element("name", "accept_cookie")
        accept_button.click()
        logger.info("De knop is aangeklikt!")
    except Exception as e:
        logger.info(f"De knop kon niet worden gevonden of aangeklikt: {e}")

    time.sleep(2)

    # Updatebare data opslag

    # NOTE - Division 2 paths,
    # 1. Search by link (pre-defined classes)
    # 2. Search by span('Toon alle') [+ scroll into view, 2e toon alle]

    all_data = []
    Path1: bool = True

    try:
        # Zoek de sectie 'product-reviews' en de knop
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//a[starts-with(@href, "/nl/productreviews/")]')
                )
            )
            driver.execute_script("arguments[0].click();", element)
        except:
            element = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "(//span[contains(text(), 'Toon alle')])[2]")
                )
            )

            # Force scroll into view using JavaScript
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'end', inline: 'nearest'});",
                element,
            )

            # Highlight the element for debugging (optional)
            driver.execute_script("arguments[0].style.border='3px solid red'", element)

            # Try clicking the element using JavaScript
            driver.execute_script("arguments[0].click();", element)
            logger.info("Element clicked via JavaScript.")
            Path1 = False
            # logger.info("De knop met tekst die begint met 'Toon alle' is aangeklikt!")
    except Exception as e:
        logger.info(
            f"Er is een fout opgetreden bij het openen van de reviews side tab: {e}"
        )

    time.sleep(2)

    # SECTION - Defining CSV Styling Function [Path 1]

    def get_content_for_path_1(element) -> dict:
        scores: list = element.find_elements(By.CLASS_NAME, "review-rating")
        titles: list = element.find_elements(By.CLASS_NAME, "reviews__item-title")
        points_list: list = element.find_elements(By.CLASS_NAME, "list")
        contents: list = element.find_elements(
            By.CLASS_NAME, "curtain__content-inner-wrapper"
        )
        bottom_contents: list = element.find_elements(
            By.CLASS_NAME, "js-inline-list-item"
        )

        pluspunten: list = []
        minpunten: list = []
        if len(points_list) > 0:
            points = points_list[0].find_elements(By.CLASS_NAME, "list__item")
            for x in points:
                icon_classes = x.find_elements(By.CLASS_NAME, "icon")
                if len(icon_classes) == 0:
                    continue

                if "green" in icon_classes[0].get_attribute("class"):
                    pluspunten.append(x.text)
                else:
                    minpunten.append(x.text)

        return {
            "score": (
                None if len(scores) == 0 else float(scores[0].text.replace(",", "."))
            ),
            "title": None if len(titles) == 0 else titles[0].text,
            "plus": ";".join(pluspunten),
            "min": ";".join(minpunten),
            "content": (
                None if len(contents) == 0 else contents[0].text.replace("\n", " ")
            ),
            "date": None if len(bottom_contents) < 1 else bottom_contents[1].text,
        }

    #!SECTION
    # SECTION - Defining CSV Styling Function [Path 2]

    def get_content_for_path_2(element) -> dict:
        score: str = element.find_element(
            By.CSS_SELECTOR, "[data-testid=review-rating-label]"
        ).text
        if "/" in score:
            score = float(score.split("/")[0].replace(",", "."))
        else:
            score = None

        title: list = element.find_elements(By.TAG_NAME, "h4")
        if len(title) > 0:
            title = title[0].text
        else:
            title

        points = element.find_element(By.TAG_NAME, "ul").find_elements(
            By.TAG_NAME, "div"
        )
        mins, plus = [], []
        if len(points) > 0:
            for p in points:
                if (
                    p.find_element(By.TAG_NAME, "svg").get_attribute("color")
                    == "#999999"
                ):
                    mins.append(p.find_element(By.TAG_NAME, "span").text)
                else:
                    plus.append(p.find_element(By.TAG_NAME, "span").text)

        content = element.find_elements(By.XPATH, "./div")
        if len(content) >= 2:
            content = content[1].text
        else:
            content = None

        d = element.find_elements(By.TAG_NAME, "ul")
        if len(d) >= 2:
            d = d[1].find_elements(By.TAG_NAME, "li")
            if len(d) >= 2:
                d = d[1].text
            else:
                d = None

        else:
            d = None

        return {
            "score": score,
            "title": title,
            "plus": ";".join(plus),
            "min": ";".join(mins),
            "content": content,
            "date": d,
        }

    all_data = []
    #!SECTION
    # SECTION - Path 1, Pre-defined classes, works fully
    if Path1:
        logger.info("Following `path 1`")
        while True:
            try:
                # Zoek alle curtain__control elementen en klik erop
                curtain_controls = driver.find_elements(
                    By.CLASS_NAME, "curtain__control"
                )
                for control in curtain_controls:
                    # Klik op elk element
                    driver.execute_script("arguments[0].click();", control)
                    logger.info("Klikken op curtain__control element")
                time.sleep(
                    2
                )  # Wacht even voor eventuele dynamische content te laden na het openen
            except Exception as e:
                logger.info(f"Fout bij het klikken op curtain__control elementen: {e}")

            # Extract de reviews content
            reviews = driver.find_elements(By.CLASS_NAME, "reviews__content-wrapper")

            # Extract de beoordelingen binnen de 'reviews__content' klasse
            data = []
            for review in reviews:
                try:
                    # Haal de reviewtekst
                    review_content = get_content_for_path_1(review)
                except Exception as e:
                    # In het geval dat een beoordeling niet gevonden kan worden, geef een default waarde
                    logger.info(f"Fout bij het ophalen van de beoordeling: {e}")

                # Voeg de review en beoordeling toe aan de data lijst
                data.append(review_content)

            # Voeg de data toe aan de all_data lijst
            all_data.extend(data)

            # Controleer op de "next page" knop
            try:
                next_button = driver.find_element(By.XPATH, '//a[@rel="next"]')
                if next_button:
                    logger.info(f"Moving to next page...")
                    next_button.click()
                    time.sleep(2)  # Wacht op de volgende pagina
                else:
                    logger.info("No more pages to load.")
                    break
            except Exception as e:
                logger.info("Geen 'next' knop gevonden, eindigen.")
                break
    #!SECTION
    # SECTION - Path 2, Pre-defined classes, works fully
    else:
        logger.info("Following path2")

        while True:
            try:

                try:
                    curtain_controls = driver.find_elements(
                        By.CLASS_NAME, "curtain__control"
                    )
                    for control in curtain_controls:
                        # Klik op elk element
                        driver.execute_script("arguments[0].click();", control)
                        logger.info("Klikken op curtain__control element")
                    time.sleep(
                        2
                    )  # Wacht even voor eventuele dynamische content te laden na het openen
                except Exception as e:
                    logger.info(
                        f"Fout bij het klikken op curtain__control elementen: {e}"
                    )

                # Extract the reviews content
                articles = driver.find_elements(
                    By.TAG_NAME, "article"
                )  # Or use By.CLASS_NAME if articles have a specific class
                # Extract text or HTML content from the articles
                data = [get_content_for_path_2(article) for article in articles]
                # data = [{"content": article.get_attribute("innerHTML")} for article in articles]
                # data = [{"content": article.text} for article in articles]

                # Add the data to the all_data list
                all_data.extend(data)

                # Check for the "next page" button

                try:
                    next_button = driver.find_element(By.XPATH, '//a[@rel="next"]')
                    if next_button:
                        logger.info(f"Moving to next page...")
                        next_button.click()
                        time.sleep(2)  # Wacht op de volgende pagina
                    else:
                        logger.info("No more pages to load.")
                        break
                except Exception as e:
                    logger.info("Geen 'next' knop gevonden, eindigen.")
                    break
            except Exception as e:
                logger.info(f"Error ophalen articles: {e}")
                break
    #!SECTION
    # Convert the data to a DataFrame
    df = pd.DataFrame(all_data)

    # Save to a CSV file
    df.drop_duplicates(inplace=True)
    df = df[df["score"].notna()]
    driver.quit()
    return df

    # time.sleep(40)
    # Sluit de driver (optioneel)
