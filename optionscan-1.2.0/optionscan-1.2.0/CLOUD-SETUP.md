# Running it in the cloud — no Python on your computer

Two ways. Start with the first one today; set up the second when you want it to
run by itself.

---

## Option A — Google Colab (works in 5 minutes, you click a button each day)

Free, runs in your browser on Google's machines, nothing installed.

1. Go to **https://colab.research.google.com**
2. **File → Upload notebook**, choose `Options_Scanner_Colab.ipynb`
3. Click the ▶ button on each cell from top to bottom. The second cell asks you
   to pick a file — choose `optionscan-1.1.0.zip`.
4. Edit the settings cell (account size, risk percent), then run the scan cells.

The report appears inside the notebook and downloads to your computer.

**What you need:** a Google account. That's all.

**The catch:** Colab forgets everything when you close it, so you re-upload the
zip each session (about 30 seconds), and the implied-volatility history does not
build up between runs — your IV rank stays as the labelled estimate rather than
becoming a true IV rank. Fine for trying it and for occasional use. For daily
use, Option B is better.

---

## Option B — GitHub Actions (runs itself every weekday, nothing to click)

Free, runs on a schedule, keeps its history, emails you if it fails. This is the
real answer for daily use. It looks like a developer tool but the whole setup is
done through the website — **you never touch a terminal.**

### 1. Make a GitHub account
**https://github.com/signup** — free.

### 2. Create a private repository
- Click the **+** (top right) → **New repository**
- Name it anything, e.g. `options-scanner`
- Choose **Private**  ← important, the reports contain your account size
- Click **Create repository**

### 3. Upload the tool
- Unzip `optionscan-1.1.0.zip` on your computer
- On the repo page click **uploading an existing file**
- Drag in **everything from inside** the unzipped folder — the `optionscan`
  folder, `requirements.txt`, `micro-500.yaml`, and the rest
- Scroll down, click **Commit changes**

> The `.github` folder is hidden on Mac and Linux. If the drag-and-drop misses
> it, create it by hand: **Add file → Create new file**, type
> `.github/workflows/daily-scan.yml` as the name (the slashes make the folders),
> paste in the contents of that file from the zip, and commit.

### 4. Set your account size
- **Settings** tab → **Secrets and variables** → **Actions** → **Variables** tab
- **New repository variable**, add these:

| Name | Value |
|---|---|
| `ACCOUNT_SIZE` | `500` |
| `RISK_PCT` | `20` |
| `UNIVERSE` | `liquid` |
| `CONFIG_FILE` | `micro-500.yaml` |

### 5. Turn it on
- **Actions** tab → if prompted, click **I understand my workflows, go ahead and
  enable them**
- Click **Daily options scan** in the sidebar → **Run workflow** → **Run
  workflow**

It runs for a couple of minutes. When the dot goes green, click into the run and
download **scan-1** from the **Artifacts** section at the bottom. That zip has
your HTML report.

From then on it runs by itself at 3:45pm Central every weekday.

### 6. Optional: get a web page instead of a download

If you would rather bookmark a URL than download a zip each day, add one more
variable: `PUBLISH_PAGES` = `true`, then **Settings → Pages → Source: GitHub
Actions**.

**Read this before you do:** GitHub Pages on a free account is **public**. The
report shows your account size and position sizes. Anyone who guesses the URL
can read it. Either leave this off and use the download, or accept that. Private
Pages requires a paid GitHub plan.

---

## Which should you use?

| | Colab | GitHub Actions |
|---|---|---|
| Setup time | 5 minutes | ~20 minutes, once |
| Runs by itself | No | Yes, every weekday |
| Builds IV-rank history | No | Yes |
| Needs | Google account | GitHub account |
| Reports stay private | Yes | Yes (unless you enable Pages) |

Use Colab today to see it work. Set up Actions this week if you intend to scan
daily — the IV-rank history alone is worth it, since that is the difference
between an estimated premium reading and a real one.

---

## Timing

Run it **after the close** (after 3pm Central) or **before the open**. Not
intraday. End-of-day quotes are settled, the tool works on daily bars, and you
get the evening to place orders instead of chasing a moving market.

---

## When something breaks

- **Actions tab shows a red X** — click the run, then the failed step. The error
  is usually a missing file from step 3 or a typo in a variable name.
- **"No trades today"** — normal, and the report lists why for each symbol. On a
  small account with a probability floor, empty days are common and correct.
- **Report says most symbols returned no data** — the scan did not really run.
  Yahoo was unreachable or rate-limited. Re-run it; the report says this
  explicitly rather than pretending the market was quiet.
