# UBS Community Insights Dashboard

Interactive demonstration dashboard for the AI for Integrated Bible Ministry workshop.

Workflow:
**Listen → Measure → Discover → Compare → Understand → Validate → Respond**

Features:
- 8 core issue scores
- Adjustable 0–100 issue threshold
- Issue combinations
- Age, Life Stage, geographic and other group analysis
- Two-group comparison
- Q41/Q42 qualitative exploration
- Pattern Discovery Worksheet
- Individual-record lists and CSV downloads

## Data notice

The included Excel workbook is **synthetic demonstration data**. It must not be presented as findings from a real church, country, denomination, or population.

Do not place real respondent data in a public deployment. For real ministry use, use an appropriately secured/private environment and follow organizational privacy, consent, safeguarding, and confidentiality requirements.

## Local run

```bash
pip install -r requirements.txt
streamlit run church_dashboard_app.py
```

## Streamlit Community Cloud

1. Create a GitHub repository, e.g. `ubs-community-insights`.
2. Upload the files in this folder to the repository.
3. In Streamlit Community Cloud, create a new app.
4. Select the repository and branch.
5. Main file: `church_dashboard_app.py`
6. Deploy.

The included Excel workbook allows the synthetic demo to load automatically.

## Workshop framing

“This is a demonstration environment for discovering patterns in synthetic community data. The dashboard helps us see where needs are concentrated; ministry leaders still interpret, validate, discern and decide what action is appropriate.”
