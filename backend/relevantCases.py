from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import faiss
import ast
from Tagger import tagger

# Load model and index
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
index = faiss.read_index("backend/legal_cases.index")

# Load and process CSV data
csv_path = "backend/flattened_output.csv"
df = pd.read_csv(csv_path)

# tag_columns = ['cat_0_value', 'cat_1_value', 'cat_2_value', 'cat_3_value', 'cat_4_value']
# df['tags'] = df[tag_columns].values.tolist()

df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

valid_tags = [
    'abetment', 'abetment-of-a-thing', 'abetment-to-suicide', 'act-to-insult-the-modesty-of-woman', 'adulteration-of-food-drink-for-sale', 
    'adverse-partys-right-to-challenge-refreshed-memory-writing', 'advocates', 'aggravated-penetrative-sexual-assault', 
    'appeal-high-court < writ-petition-high-court', 'appeal-to-high-court, high-court-power-of-revisions', 'appeal-to-supreme-court, power-to-issue-writ', 
    'application-to-magistrate', 'arms', 'army', 'arrest-without-warrant', 'assault-on-public-servant', 'assault-on-woman', 
    'assisting-in-concealment-of-stolen-property', 'attachment-of-property-for-proclaimed-person', 'attempt-to-murder', 'bail-in-non-bailable-offence', 
    'banking-regulation', 'bar-of-limitation', 'bihar-vat', 'board, securities', 'breach-of-peace', 'breach-of-trust', 'breach-of-trust-by-clerk-servant', 
    'burden-of-proof-for-known-facts', 'burden-of-proving-exception-in-criminal-cases', 'buying-a-slave', 'calling-for-records, examine-the-records', 
    'cause-hurt', 'causing-explosion-to-endanger-life-or-property', 'causing-hurt-to-public-servant', 'causing-hurt-while-committing-robbery', 
    'central-state-laws', 'cheating', 'cheque-dishonour', 'child-marriage < legitimacy-of-children < prohibition-of-child-marriage < custody-and-maintenance', 
    'children-protection', 'cognizance-of-offences', 'common-intention', 'companies-law', 'compounding-of-offences', 'conditions-for-search-of-persons', 
    'conduct-of-business-of-government-of-a-state < all-executive-action-shall-be-expressed-in-the-name-of-the-governor', 'confession-recording, recording-of-statement', 
    'conspirators-statements-and-actions', 'contempt-of-court, fair-and-innocent-publication', 'cooperative-societies', 'corruption', 'court-ordered-questioning-and-witness-compulsion', 
    'court-superintendence', 'criminal-breach-of-trust', 'criminal-conspiracy', 'criminal-intimidation', 'criminal-law, act-to-supplement-the-criminal-law', 
    'criminal-misconduct', 'criminal-misconduct-by-public-servant', 'criminal-procedure-code', 'cross-examining-of-written-statements', 'culpable-homicide', 'death-by-negligence', 
    'defamation', 'definition', 'definition-murder', 'delhi-special-police, police-act', 'destruction-of-evidence', 'detention-period-undergone-negated-from-final-sentence', 
    'directions-for-grant-of-bail-to-person-apprehending-arrest', 'dismissal-removal-reduction-of-persons-employed', 'disobedience', 'disposal-of-property', 'domestic-violence', 
    'double-conviction', 'dowry', 'dowry-as-criminal-offence', 'dowry-death', 'electricity-regulations', 'employer-state-insurance', 'endangering-life-or-personal-safety, negligent-rash-act', 
    'equal-opportunity', 'equality-before-law', 'evidence-by-refusal-of-production', 'examination-of-complainant', 'examination-of-witnesses', 'examine-accused', 
    'explosive-substances', 'extension-of-prescribed-period', 'false-evidence', 'false-property-mark', 'false-statement-in-verification', 'federal-nature, parliament-vs-state-assembly', 
    'foreign-exchange-and-smuggling-prevention', 'forged-document', 'forgery', 'free-speech', 'freedom-with-reasonable-restrictions', 'general-clauses', 'gratification-other-than-legal-remuneration', 
    'grievous-hurt', 'harbouring-offender', 'house-trespass', 'house-trespass-punishable-with-imprisonment', 'income-tax', 'indecent-and-scandalous-questions', 'indian-penal-code', 
    'indian-stamp, use-of-stamps', 'individual-freedom', 'information-in-cognizable-cases', 'injuring-or-defiling-place-of-worship', 'interpretation-clause', 'investigation-of-affairs-of-a-company', 
    'kidnapping', 'kidnapping, abducting', 'kidnapping-abducting-for-murder', 'kidnapping-for-ransom', 'kidnapping-to-compel-marriage', 'law-for-consumers', 'law-governing-customs', 
    'law-governing-evidence', 'law-governing-narcotics', 'law-regarding-motor-vehicles', 'legal-and-illegal-considerations-and-objects', 'licence-for-acquisition-and-possession, firearms-and-ammunition', 
    'licence-for-acquisition-of-arms', 'limitation-executive', 'lurking-house-trespass-punishable-with-imprisonment', 'magistrate-empowered-under-section-190-to-order-investigation', 
    'magistrate-may-order-detention', 'magistrates-take-cognizance-of-offences', 'miscarriage-without-consent', 'mischief', 'mischief-by-fire-with-intent-to-destroy-house', 
    'murder', 'national-security', 'negotiable-instruments', 'non-appearance', 'non-bailable-offences', 'nothing-to-preclude-further-investigation', 'obstructing-public-servant', 
    'offences-against-public-justice', 'order-enforcement', 'order-prosecution', 'partnership,indian-partners,', 'passport', 'penalty-for-demanding-dowry, punishment-for-dowry', 
    'penetrative-sexual-assault', 'police-report-on-completion-of-investigation', 'police-to-enquire-and-report-on-suicide', 'postponement-of-issue-of-process', 'power-high-court-for-quashing', 
    'power-of-police-to-seize-property', 'power-to-proceed-against-other-persons-appearing', 'powers-without-warrant', 'preparation-to-dacoity, commit-dacoity', 
    'presumption-for-law-collections-and-decision-reports', 'preventive-detention', 'previous-sanction-necessary-for-prosecution', 'probation-of-offenders, criminal', 
    'proclamation-for-absconding-persons', 'procuration-of-minor-girl', 'prohibition-of-discrimination', 'prosecution-of-judges-and-public-servants', 'prostitution', 
    'protection-against-arrests-and-detention', 'protection-of-life-and-liberty', 'punishment-for-aggravated-penetrative-sexual-assault', 'punishment-for-assault-without-provocation', 
    'punishment-for-attempt-to-cause-explosion', 'punishment-for-cheating', 'punishment-for-cheating-by-impersonation', 'punishment-for-contempt-of-court', 'punishment-for-criminal-trespass', 
    'punishment-for-defamation', 'punishment-for-earning-from-prostitution', 'punishment-for-extortion', 'punishment-for-house-trespass', 'punishment-for-keeping-a-brothel', 
    'punishment-for-penetrative-sexual-assault', 'punishment-for-prohibited-arms', 'punishment-for-using-arms', 'questions-intended-to-insult-or-annoy', 'rape', 'rash-and-negligent-driving', 
    're-marriage-during-husband-or-wife-lifetime', 'reasonable-grounds-for-questioning', 'release-offenders-on-probation-for-good-conduct', 'relevancy-of-entry-in-electronic-public-record', 
    'repealed', 'repealed-by-prevention-of-corruption-act', 'restitution-of-conjugal-right', 'right-to-information', 'rioting', 'robbery', 'scheduled castes-scheduled-tribes, sc-st < prevention-of-atrocities', 
    'special-leave-appeal-by-supreme-court', 'special-power-of-high-court < power-of-high-court-regarding-bail', 'statements-to-police-not-to-be-signed', 'theft', 'time-exclusion-from-limitation', 
    'transfer-by-unauthorized-person-with-later-acquisition-of-interest', 'transfer-of-property', 'transfer-of-property-defined', 'unlawful-activities-uapa', 'unlawful-assembly', 
    'unnatural-offences', 'voluntarily-causing-hurt', 'when-to-discharge-accused', 'who-may-testify', 'wilful-attempt-to-evade-tax', 'will-forgery', 'witness-immunity-in-criminal-proceedings', 
    'wrongful-confinement', 'wrongful-restraint', 'wrongfully-concealed-kidnapping'
]

def Case_search(query, tags=[], top_k=5):
    # Sanitize and filter tags
    tags = [tag.strip() for tag in tags if tag.strip() in valid_tags]
    if not tags:
        print("No valid tags provided.")
        return None
    print(f"Tags used in search: {tags}") 
    # Step 1: Filter dataframe based on tag matches
    def tag_filter(row):
        # Check if any of the tag columns contain any of the provided tags
        for col in ['cat_0_value', 'cat_1_value', 'cat_2_value', 'cat_3_value', 'cat_4_value']:
            if any(tag in row[col] for tag in tags):
                return True
        return False

    filtered_df = df[df.apply(tag_filter, axis=1)].copy()
    if filtered_df.empty:
        print("No cases found for the given tags.")
        return None

    # Print directly fetched cases from the CSV (tag-filtered)
    print("\n=== Cases fetched directly from CSV (tag filtered) ===")
    print(filtered_df[['title', 'tags']])

    # Step 2: Semantic FAISS search on full query
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), len(df))  # Search across full index

    # Map FAISS index to DataFrame indices
    faiss_results = df.iloc[I[0]].copy()
    faiss_results['faiss_rank'] = range(len(faiss_results))

    # Print cases fetched by FAISS
    print("\n=== Cases fetched by FAISS ===")
    print(faiss_results[['title', 'tags']])

    # Step 3: Merge tag-filtered cases with FAISS results to retain only relevant ones
    merged = pd.merge(filtered_df, faiss_results[['title', 'faiss_rank']], on='title', how='inner')

    # Step 4: Sort by FAISS rank to prioritize semantic relevance
    merged = merged.sort_values(by='faiss_rank').head(top_k)

    # Print merged results (final output)
    print("\n=== Merged and ranked cases ===")
    print(merged[['title', 'tags']])

    return merged


examples = [
    {
        "query": "charges for theft?",
        "tags": tagger("charges for theft?", top_k=5, min_score=0.4)
    }
]

for idx, example in enumerate(examples, start=1):
    print(f"\n=== Example {idx}: {example['query']} ===")
    results = Case_search(
        query=example['query'],
        tags=example['tags'],
        top_k=5
    )
    if results is not None:
        print(results[['title', 'tags']])  # Printing the title and tags
    else:
        print("No matching results.")
