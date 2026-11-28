# Real price data

`lemon_daily_agmarknet.csv.gz` holds daily lemon prices from Agmarknet (https://agmarknet.gov.in/), India's government mandi price records: 157,829 rows across 667 mandis in 23 states, 1 January 2022 to 23 November 2024.

**Where the file came from.** We took it from a public extract on GitHub: `LemonPrices.csv` in
https://github.com/Aash002/Lemon-Price-Analysis-Case-Study. The columns were renamed (`Modal Price(Rs./Quintal)` to `modal`, `Arrivals (Tonnes)` to `arrivals`), rows with no date, market, state or price, or a zero price, were dropped, and the rest saved here with a `commodity` column added.

The prices themselves are Agmarknet's, published under the Government Open Data Licence India. The GitHub repository that hosts the extract has no licence file of its own.

`calibration.json` is produced from this file by `python experiments/calibrate_real.py`.
