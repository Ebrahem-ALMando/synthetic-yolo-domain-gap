import fs from "node:fs";
import path from "node:path";
import puppeteer from "puppeteer-core";

const executablePath =
  process.env.CHROME_PATH ?? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const baseUrl = process.env.DASHBOARD_URL ?? "http://127.0.0.1:3106";
const allowMissingLogo = process.env.ALLOW_MISSING_OFFICIAL_LOGO === "1";
const screenshotRoot = path.resolve(
  process.cwd(),
  process.env.DASHBOARD_SCREENSHOT_DIR ?? "../../artifacts/dashboard_previews/final-sprint6a",
);
const widths = [390, 768, 1024, 1280, 1440];
const routes = [
  "/",
  "/experiments",
  "/experiments/synthetic_only",
  "/datasets/real",
  "/datasets/synthetic",
  "/training",
  "/evaluation",
  "/analysis",
  "/inference",
  "/reports",
  "/reproducibility",
  "/system",
  "/about",
];

if (!fs.existsSync(executablePath)) throw new Error(`Chrome not found: ${executablePath}`);
fs.mkdirSync(screenshotRoot, { recursive: true });

const browser = await puppeteer.launch({ executablePath, headless: true });
const failures = [];
try {
  for (const width of widths) {
    for (const route of routes) {
      const page = await browser.newPage();
      await page.setViewport({ width, height: 900, deviceScaleFactor: 1 });
      const consoleErrors = [];
      const pageErrors = [];
      page.on("console", (message) => {
        if (message.type() === "error") consoleErrors.push(message.text());
      });
      page.on("pageerror", (error) => pageErrors.push(error.message));
      const response = await page.goto(`${baseUrl}${route}`, { waitUntil: "networkidle0" });
      await page.waitForSelector("main");

      let mobileNavigationOpened = true;
      let themeToggled = true;
      if (route === "/experiments") {
        if (width < 1024) {
          await page.click('button[aria-label="فتح التنقل"]');
          mobileNavigationOpened =
            (await page.$('[role="dialog"] nav[aria-label="التنقل الرئيسي"]')) !== null;
          await page.keyboard.press("Escape");
        }
        const themeBefore = await page.evaluate(() =>
          document.documentElement.classList.contains("dark"),
        );
        const themeSelector = themeBefore
          ? 'button[aria-label="تفعيل الوضع الفاتح"]'
          : 'button[aria-label="تفعيل الوضع الداكن"]';
        await page.waitForSelector(themeSelector);
        await page.click(themeSelector);
        const themeAfter = await page.evaluate(() =>
          document.documentElement.classList.contains("dark"),
        );
        themeToggled = themeBefore !== themeAfter;
      }

      const result = await page.evaluate(() => {
        const viewport = document.documentElement.clientWidth;
        const overflow = [...document.querySelectorAll("body *")]
          .map((element) => {
            const rect = element.getBoundingClientRect();
            const intentionalScroll = [...document.querySelectorAll(".overflow-x-auto")].some(
              (container) => container !== element && container.contains(element),
            );
            return {
              tag: element.tagName,
              className:
                typeof element.className === "string" ? element.className.slice(0, 120) : "",
              left: Math.round(rect.left),
              right: Math.round(rect.right),
              width: Math.round(rect.width),
              intentionalScroll,
            };
          })
          .filter(
            (item) =>
              (item.left < -1 || item.right > viewport + 1) && !item.intentionalScroll,
          )
          .sort((a, b) => b.width - a.width)
          .slice(0, 8);
        const logo = document.querySelector('img[alt*="الشعار الرسمي"]');
        return {
          viewport,
          scrollWidth: document.documentElement.scrollWidth,
          dir: document.documentElement.dir,
          lang: document.documentElement.lang,
          overflow,
          logoLoaded:
            logo instanceof HTMLImageElement && logo.complete && logo.naturalWidth > 0,
          mobileMenuVisible:
            viewport >= 1024 || Boolean(document.querySelector('button[aria-label="فتح التنقل"]')),
          geometry: {
            body: Math.round(document.body.getBoundingClientRect().width),
            main: Math.round(document.querySelector("main")?.getBoundingClientRect().width ?? 0),
            mainLeft: Math.round(
              document.querySelector("main")?.getBoundingClientRect().left ?? 0,
            ),
          },
        };
      });

      const routeName = route === "/" ? "overview" : route.slice(1).replaceAll("/", "-");
      await page.screenshot({
        path: path.join(screenshotRoot, `${width}-${routeName}.png`),
        fullPage: true,
      });

      if (!response || response.status() >= 400)
        failures.push({ width, route, kind: "http", status: response?.status() });
      if (result.scrollWidth > result.viewport || result.overflow.length)
        failures.push({ width, route, kind: "overflow", ...result });
      if (result.dir !== "rtl" || result.lang !== "ar")
        failures.push({ width, route, kind: "rtl", ...result });
      if (!result.mobileMenuVisible)
        failures.push({ width, route, kind: "mobile_navigation", ...result });
      if (!mobileNavigationOpened)
        failures.push({ width, route, kind: "mobile_navigation_interaction" });
      if (!themeToggled) failures.push({ width, route, kind: "theme_interaction" });
      if (!result.logoLoaded && !allowMissingLogo)
        failures.push({ width, route, kind: "official_logo_missing" });
      if (consoleErrors.length && !allowMissingLogo)
        failures.push({ width, route, kind: "console", consoleErrors });
      if (pageErrors.length) failures.push({ width, route, kind: "pageerror", pageErrors });
      console.log(
        JSON.stringify({
          width,
          route,
          status: response?.status(),
          ...result,
          mobileNavigationOpened,
          themeToggled,
          consoleErrors: consoleErrors.length,
          pageErrors: pageErrors.length,
        }),
      );
      await page.close();
    }
  }
} finally {
  await browser.close();
}

if (failures.length) {
  console.error(JSON.stringify({ status: "failed", failures }, null, 2));
  process.exit(1);
}
console.log(
  JSON.stringify({
    status: "passed",
    routeCount: routes.length,
    viewportCount: widths.length,
    screenshotRoot,
  }),
);
