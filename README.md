# Review Quality Classifier

Multimodal pipeline that flags low-quality Google location reviews (advertisements, irrelevant content, rants from non-visitors) so that moderators only need to look at the ambiguous ones.

Built in 4 weeks by a team of five for **TikTok TechJam 2025**.

<!-- TODO: add one screenshot or diagram of the pipeline here. A single image raises the perceived quality of the repo more than any paragraph. -->

## The problem

Review platforms have no labelled data for "bad" reviews, and manual moderation does not scale. We needed a classifier that could:

- sort reviews into four classes: `Advertisement`, `Irrelevant`, `Rant Without Visit`, `Valid Review`
- use both the review text and the attached photo
- be trained without a labelled dataset

## Approach

| Stage | What it does | Tools |
|---|---|---|
| 1. Image captioning | Turns each review photo into a text caption and merges it with the cleaned review text | BLIP (Hugging Face Transformers) |
| 2. Feature engineering | Represents each review as keyword features plus a semantic embedding | TF-IDF, Sentence-Transformers |
| 3. Pseudo-labelling, rule stage | Labels the obvious cases with high-confidence rules | Pandas |
| 4. Pseudo-labelling, model stage | Trains a baseline on the rule-labelled rows, then labels the remaining ambiguous rows | scikit-learn `LogisticRegression` |
| 5. Final classifier | Stacking ensemble trained on the fully labelled set | scikit-learn, LightGBM |

**Why two-stage pseudo-labelling:** rules alone are precise but cover few reviews; a model alone has nothing to learn from. Using rules to seed a baseline model gave us labels for the whole dataset without manual annotation.

## Results

| Metric | Value |
|---|---|
| Overall accuracy | 88% |
| Test set size | <!-- TODO: number of hand-labelled reviews --> |
| Per-class precision / recall | <!-- TODO: paste the classification report from the notebook --> |

<!-- TODO: add the confusion matrix image exported from the notebook. -->

### How to read the 88%

The test set was labelled by hand by the team, and we deliberately picked reviews that were clearly valid or clearly violating. This means:

- the score reflects performance on **clear-cut cases**, and is likely higher than it would be on a random sample of reviews
- labels were assigned by team judgement, without a written rubric or a second annotator

## Limitations and what we would do next

- **Evaluate on a random sample** with a written labelling rubric and two annotators, and report agreement between them.
- **Report per-class metrics** rather than accuracy alone, since valid reviews heavily outnumber violations.
- **Measure label noise** from the pseudo-labelling stages by spot-checking a sample of the generated labels.
- **Add unit tests for the rule stage**, since every downstream label depends on those rules being correct.
- **Package the pipeline as a script** so it runs outside Google Colab.

## Repository structure

```
notebooks/Main_Pipeline.ipynb   Full training and evaluation pipeline
data/                           Input data (not included, see below)
requirements.txt                Python dependencies
```

<!-- TODO: check this matches the actual repo layout. -->

## Running it

The notebook was written for Google Colab with data stored in Google Drive.

1. Clone the repository:
   ```bash
   git clone https://github.com/Capacesse/review-quality-classifier.git
   cd review-quality-classifier
   ```
2. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Add a Hugging Face token to Colab Secrets under the name `TikTokTechJam2025`.
4. Place `merged_reviews_4.csv` and the `dataset/` image folder in `data/`:
   ```
   data/
   ├── merged_reviews_4.csv
   └── dataset/
       ├── indoor_atmosphere/
       ├── menu/
       ├── outdoor_atmosphere/
       └── taste/
   ```
5. Open `notebooks/Main_Pipeline.ipynb` in Colab and run all cells.

The dataset is not included in this repository. <!-- TODO: say where it came from, or link the source. -->

## Team

| Member | Role |
|---|---|
| Julius Ng Hwee Chong | Project lead: scoped the problem, designed the pipeline architecture, coordinated integration |
| Phoa Li Lynn Sheryl | Data preprocessing |
| Park Soohwan | Feature engineering |
| Danny Lee Han Keat | Model development and pseudo-labelling |
| Mahesh Sanjana | Policy and evaluation |
