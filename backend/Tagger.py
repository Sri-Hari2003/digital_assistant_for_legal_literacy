from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import json
import os

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

tag_embeddings = embedding_model.embed_documents(valid_tags)

def tagger(user_query: str, top_k: int = 3, min_score: float = 0.4):
    # Convert user query into an embedding
    query_embedding = embedding_model.embed_query(user_query)

    # Compute cosine similarity between query and each tag
    similarities = cosine_similarity([query_embedding], tag_embeddings)[0]

    # Zip tags with their scores, sort by similarity
    tag_scores = sorted(zip(valid_tags, similarities), key=lambda x: x[1], reverse=True)

    # Filter by min_score and pick top_k
    filtered_tags = [tag for tag, score in tag_scores if score >= min_score][:top_k]

    return filtered_tags

def search_tags_in_csv(tags):
    path = r"C:\Users\S Sri Hari\Major_project\final\frontend\chatbot-ui\backend\flattened_output.csv"
    df = pd.read_csv(path)

    # Columns containing tags
    tag_columns = [col for col in df.columns if col.startswith('cat_') and col.endswith('_value')]

    # Count how many tags match per row
    df['match_count'] = df[tag_columns].apply(lambda row: sum(tag in tags for tag in row), axis=1)

    # Filter rows with at least one match
    matching_rows = df[df['match_count'] > 0]

    # Sort by most matched tags
    matching_rows = matching_rows.sort_values(by='match_count', ascending=False)

    if matching_rows.empty:
        return pd.DataFrame()  # No matches at all

    max_match = matching_rows['match_count'].max()

    if max_match >= 3:
        final_rows = matching_rows[matching_rows['match_count'] == max_match]
    elif max_match == 2:
        final_rows = matching_rows[matching_rows['match_count'] == 2].head(5)
    elif max_match == 1:
        final_rows = matching_rows[matching_rows['match_count'] == 1].head(3)
    else:
        final_rows = pd.DataFrame()

    return final_rows[['title', 'docid', 'position', 'match_count']]


# Example usage
# tags = ['theft', 'assisting-in-concealment-of-stolen-property', 'robbery', 'punishment-for-extortion', 'punishment-for-criminal-trespass']
# print(search_tags_in_csv(tags))

def search_and_load_json(tags, root_folder):
    matched_rows = search_tags_in_csv(tags)

    results = []

    for _, row in matched_rows.iterrows():
        position = str(row['position'])
        docid = str(row['docid'])
        json_path = os.path.join(root_folder, position, f"{docid}.json")

        # Load JSON if it exists
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            results.append({
                'title': row['title'],
                # 'docid': docid,
                # 'position': position,
                'match_count': row['match_count'],
                "content": content.get("doc", "")
            })
        else:
            print(f"[!] File not found: {json_path}")

    return results


# tags = ['theft', 'assisting-in-concealment-of-stolen-property', 'robbery', 'punishment-for-extortion', 'punishment-for-criminal-trespass']
# root_folder = r"C:\Users\S Sri Hari\Major_project\final\frontend\chatbot-ui\backend\criminal law -IPC"

# results = search_and_load_json(tags, root_folder)

# for res in results:
#     print(f"Title: {res['title']}")
#     print(f"Doc ID: {res['docid']}")
#     print(f"Position: {res['position']}")
#     print(f"Matched Tags: {res['match_count']}")
#     print(f"Excerpt: {res['content'][:300]}\n")
