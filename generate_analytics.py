#!/usr/bin/env python3
"""
Build charts from my own public GitHub activity and refresh the README.

Outputs (written to ./assets):
  - commit-heatmap.png     commits by weekday x hour (local time)
  - domain-breakdown.png   commits per analytics domain
  - language-mix.png       share of code by language across my repos

It also rewrites the block between <!-- ANALYTICS:START --> and
<!-- ANALYTICS:END --> in README.md with the charts and a few auto-generated insights.

Run locally:   GITHUB_TOKEN=ghp_xxx python scripts/generate_analytics.py
Run in CI:     see .github/workflows/analytics.yml
"""

import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (works in GitHub Actions)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
USERNAME = "T3MP35TT"
TIMEZONE = "Asia/Kolkata"  # commit timestamps are converted to this timezone
MAX_COMMIT_PAGES = 5  # 5 pages x 100 = up to 500 commits per repo
TOP_LANGUAGES = 8

ASSETS = Path("assets")
README = Path("README.md")

# Repo name -> domain. Repos not listed here are ignored in the domain chart.
DOMAINS = {
    "Business & Marketing": [
        "Sugar-Cosmetics-Instagram-Analytics",
        "Shopify-Sales-and-Customer-Behaviour-Analysis",
        "Coffee-Shop-Sales-Analysis-Power-BI",
        "Amazon-Customer-Behavior-Analysis",
        "CustomerIQ-AI-Retention-Analytics",
    ],
    "Mobility & Logistics": [
        "OLA-Ride-Insights",
        "Advanced-Data-Visualisation-Analysis-on-Namma-Yatri-Data",
        "EDA_Optimising_NYC_Taxis",
        "Delivery-Time-Estimation-Using-Linear-Regression",
    ],
    "Finance & Real Estate": [
        "Real-Estate-Analytics-and-Forecasting",
        "Stock-Market-Analysis",
        "Stock-Market-Portfolio-Risk-Analytics",
    ],
    "People & Workforce": [
        "Predicting-Employee-Retention",
        "Mental-Health-In-Tech-Analysis",
    ],
    "Sports & Fitness": [
        "Cricbuzz-LiveStats-Cricket-Analytics-Dashboard",
        "Strava-Fitness-Analytics",
    ],
    "Environment": [
        "Bird-Species-Observation-Analysis",
    ],
}
REPO_TO_DOMAIN = {repo.lower(): d for d, repos in DOMAINS.items() for repo in repos}

# Chart styling (matches the github-dark / tokyonight look of the README)
BG = "#0d1117"
FG = "#c9d1d9"
MUTED = "#8b949e"
GRID = "#30363d"
ACCENT = "#2f81f7"
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "gh_blue", ["#161b22", "#0c2d6b", "#1158c7", "#2f81f7", "#79c0ff"]
)

API = "https://api.github.com"


# ----------------------------------------------------------------------------
# GitHub API helpers
# ----------------------------------------------------------------------------
def make_session():
    s = requests.Session()
    s.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": f"{USERNAME}-profile-analytics",
        }
    )
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    else:
        print("Warning: no GITHUB_TOKEN set, using unauthenticated requests (60/hour limit).")
    return s


def get_paginated(session, url, params=None, max_pages=10):
    results = []
    params = dict(params or {})
    params["per_page"] = 100
    for page in range(1, max_pages + 1):
        params["page"] = page
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 409:  # empty repository
            return results
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        results.extend(data)
        if len(data) < 100:
            break
    return results


def fetch_repos(session):
    repos = get_paginated(session, f"{API}/users/{USERNAME}/repos", {"type": "owner"})
    return [r["name"] for r in repos if not r["fork"]]


def fetch_commits(session, repo_names):
    rows = []
    for name in repo_names:
        commits = get_paginated(
            session,
            f"{API}/repos/{USERNAME}/{name}/commits",
            {"author": USERNAME},
            max_pages=MAX_COMMIT_PAGES,
        )
        for c in commits:
            rows.append({"repo": name, "date": c["commit"]["author"]["date"]})
    df = pd.DataFrame(rows, columns=["repo", "date"])
    if not df.empty:
        df["local"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(TIMEZONE)
        df["domain"] = df["repo"].str.lower().map(REPO_TO_DOMAIN)
    return df


def fetch_languages(session, repo_names):
    totals = Counter()
    for name in repo_names:
        r = session.get(f"{API}/repos/{USERNAME}/{name}/languages", timeout=30)
        r.raise_for_status()
        totals.update(r.json())
    return totals


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def style_axes(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)


def new_figure(figsize, title, subtitle=None):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    style_axes(ax)
    fig.text(0.02, 0.96, title, color=FG, fontsize=13, fontweight="bold", ha="left", va="top")
    if subtitle:
        fig.text(0.02, 0.89, subtitle, color=MUTED, fontsize=9, ha="left", va="top")
    return fig, ax


def save(fig, path):
    fig.savefig(path, dpi=160, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def chart_heatmap(df, path):
    grid = np.zeros((7, 24), dtype=int)
    for wd, hr in zip(df["local"].dt.weekday, df["local"].dt.hour):
        grid[wd, hr] += 1

    fig, ax = new_figure(
        (11, 3.8),
        "When I commit",
        f"Commits by weekday and hour ({TIMEZONE}) · {len(df)} commits",
    )
    fig.subplots_adjust(top=0.78)
    im = ax.imshow(grid, cmap=HEAT_CMAP, aspect="auto")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], color=FG, fontsize=9)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], color=FG, fontsize=9)
    # thin gaps between cells
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color=BG, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    save(fig, path)
    return grid


def chart_domains(df, path):
    counts = df["domain"].dropna().value_counts().sort_values()
    fig, ax = new_figure(
        (6, 3.6), "Where my work goes", "Commits per analytics domain"
    )
    fig.subplots_adjust(top=0.78, left=0.33)  # leave room for the y labels so the title lines up
    bars = ax.barh(counts.index, counts.values, color=ACCENT, height=0.6)
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_width() + max(counts.values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            color=FG,
            fontsize=9,
        )
    ax.set_xticks([])
    ax.tick_params(axis="y", colors=FG, labelsize=9)
    ax.set_xlim(0, max(counts.values) * 1.15)
    save(fig, path)
    return counts


def chart_languages(totals, path):
    total_bytes = sum(totals.values())
    ranked = totals.most_common()
    top = ranked[:TOP_LANGUAGES]
    other = sum(v for _, v in ranked[TOP_LANGUAGES:])
    if other:
        top.append(("Other", other))
    names = [n for n, _ in top][::-1]
    pct = [v / total_bytes * 100 for _, v in top][::-1]

    fig, ax = new_figure(
        (6, 3.6), "What I write in", "Share of code (by bytes) across my public repos"
    )
    fig.subplots_adjust(top=0.78, left=0.33)  # leave room for the y labels so the title lines up
    bars = ax.barh(names, pct, color=ACCENT, height=0.6)
    for bar, value in zip(bars, pct):
        ax.text(
            bar.get_width() + max(pct) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            color=FG,
            fontsize=9,
        )
    ax.set_xticks([])
    ax.tick_params(axis="y", colors=FG, labelsize=9)
    ax.set_xlim(0, max(pct) * 1.15)
    save(fig, path)
    return ranked


# ----------------------------------------------------------------------------
# README update
# ----------------------------------------------------------------------------
def fmt_hour(h):
    return f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}"


def update_readme(n_commits, n_repos, busiest_day, busiest_hour, top_domain, top_language):
    if not README.exists():
        print("README.md not found, skipping README update.")
        return
    text = README.read_text(encoding="utf-8")
    pattern = re.compile(r"<!-- ANALYTICS:START -->.*?<!-- ANALYTICS:END -->", re.S)
    if not pattern.search(text):
        print("Markers not found in README.md, skipping README update.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    insights = [f"- 🕘 Most active day: **{busiest_day}**, busiest hour: **{fmt_hour(busiest_hour)}**"]
    if top_domain:
        insights.append(f"- 🏆 Most-worked domain: **{top_domain}**")
    if top_language:
        insights.append(f"- 🐍 Top language by code volume: **{top_language}**")

    block = "\n".join(
        [
            "<!-- ANALYTICS:START -->",
            f'<p align="center"><img src="assets/commit-heatmap.png?v={stamp}" alt="Commit heatmap by weekday and hour" width="100%" /></p>',
            '<p align="center">',
            f'  <img src="assets/domain-breakdown.png?v={stamp}" alt="Commits per analytics domain" width="49%" />',
            f'  <img src="assets/language-mix.png?v={stamp}" alt="Language mix" width="49%" />',
            "</p>",
            "",
            f"> 🤖 **Auto-generated** from {n_commits} commits across {n_repos} public repos · refreshed {stamp}",
            ">",
            *[f"> {line}" for line in insights],
            "<!-- ANALYTICS:END -->",
        ]
    )
    README.write_text(pattern.sub(lambda m: block, text), encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ASSETS.mkdir(exist_ok=True)
    session = make_session()

    repos = fetch_repos(session)
    print(f"Found {len(repos)} public, non-fork repos.")

    commits = fetch_commits(session, repos)
    if commits.empty:
        print("No commits found, nothing to chart.")
        return 0
    print(f"Fetched {len(commits)} commits.")

    grid = chart_heatmap(commits, ASSETS / "commit-heatmap.png")
    domain_counts = chart_domains(commits, ASSETS / "domain-breakdown.png") if commits["domain"].notna().any() else pd.Series(dtype=int)
    languages = fetch_languages(session, repos)
    ranked = chart_languages(languages, ASSETS / "language-mix.png") if languages else []

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    busiest_day = day_names[int(grid.sum(axis=1).argmax())]
    busiest_hour = int(grid.sum(axis=0).argmax())
    top_domain = domain_counts.idxmax() if len(domain_counts) else None
    top_language = ranked[0][0] if ranked else None

    update_readme(len(commits), len(repos), busiest_day, busiest_hour, top_domain, top_language)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
