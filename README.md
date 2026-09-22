# RayX.co.uk (static site)

One-page site for RayX — a trustworthy measurement layer for farmland, starting with crop yield.

Sections: why it matters (2026 season timeline) · what we have (live services: Field Passports → passport.rayx.co.uk, GB wheat forecast → wheat.rayx.co.uk, US maize forecast → maize.rayx.co.uk, data & API) · why trust it · what we're looking for · about · contact.

- `index.html` — the page
- `assets/css/style.css` — styles
- `assets/img/` — wordmark (`rayx-wordmark.svg`), hero map, product screenshots (WebP, 1600 + 900 px) and the social share image `og-rayx.png`

Figures on the page are dated outputs from the live services — refresh them each season.

## Preview locally
Paths are root-relative, so serve the folder rather than opening the file:

    python3 -m http.server 8000   # then open http://localhost:8000

## Deploy
GitHub Pages from `main` / root (custom domain in `CNAME`). Commit and push to `main`.
