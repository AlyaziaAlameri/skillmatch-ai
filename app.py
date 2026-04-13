import ast
import re
from difflib import SequenceMatcher
from urllib.parse import quote

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillMatch AI", page_icon="🎯", layout="wide")

# =====================================================
# DATA
# =====================================================

def parse_skills(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


@st.cache_data
def load_jobs() -> pd.DataFrame:
    df = pd.read_csv("uae_jobs_with_skills.csv")
    for col in ["title", "source", "location", "url", "description", "skills"]:
        if col not in df.columns:
            df[col] = ""
    df["skills"] = df["skills"].apply(parse_skills)
    df["title"] = df["title"].fillna("Untitled Role")
    df["source"] = df["source"].fillna("Unknown Source")
    df["location"] = df["location"].fillna("UAE")
    df["url"] = df["url"].fillna("")
    df["description"] = df["description"].fillna("")
    return df


JOBS_DF = load_jobs()

# =====================================================
# SKILL & LEARNING MAPS
# =====================================================

SKILL_DICT = {
    "python": ["python", "py", "بايثون"],
    "sql": ["sql", "structured query language", "mysql", "postgresql", "database"],
    "excel": ["excel", "spreadsheets", "xls", "xlsx", "إكسل", "اكسل"],
    "power bi": ["power bi", "powerbi", "dashboards", "bi reports"],
    "pandas": ["pandas"],
    "statistics": ["statistics", "statistical", "probability", "hypothesis testing"],
    "data analysis": ["data analysis", "analyze data", "analytics", "data analyst", "analysis"],
    "data visualization": ["data visualization", "visualization", "charts", "dashboards", "reporting"],
    "machine learning": ["machine learning", "ml", "predictive modeling"],
    "artificial intelligence": ["artificial intelligence", "ai", "llm", "nlp"],
    "project management": ["project management", "project planning", "دارة المشاريع", "agile", "scrum"],
    "communication": ["communication", "communications", "presentation", "written communication", "verbal communication"],
    "leadership": ["leadership", "team lead", "leading teams"],
    "cloud": ["cloud", "aws", "azure", "gcp", "cloud computing"],
    "cybersecurity": ["cybersecurity", "information security", "soc", "siem", "pentest", "security operations"],
    "networking": ["network", "networking", "tcp/ip", "routing", "switching"],
    "accounting": ["accounting", "accountant", "bookkeeping"],
    "finance": ["finance", "financial analysis", "budgeting", "forecasting"],
    "auditing": ["audit", "auditing", "internal audit"],
    "procurement": ["procurement", "purchasing", "vendor management"],
    "testing": ["testing", "qa", "quality assurance", "test cases"],
    "software development": ["software development", "software engineering", "programming", "coding", "development"],
    "technical support": ["technical support", "it support", "help desk", "service desk", "desktop support"],
    "marketing": ["marketing", "digital marketing", "content marketing", "seo"],
    "sales": ["sales", "business development", "client acquisition", "lead generation"],
    "human resources": ["human resources", "hr", "recruitment", "talent acquisition"],
    "compliance": ["compliance", "regulatory", "controls"],
    "risk management": ["risk", "risk management", "risk assessment"],
    "operations": ["operations", "operational", "process improvement"],
}

NOISY_SKILLS = {"operations", "human resources", "finance", "testing", "communication", "cloud"}

LEARNING_RESOURCES = {
    "python": {
        "title": "Kaggle Learn: Python",
        "provider": "Kaggle",
        "url": "https://www.kaggle.com/learn/python",
        "time": "~5 hours",
        "type": "Hands-on course",
    },
    "pandas": {
        "title": "Kaggle Learn: Pandas",
        "provider": "Kaggle",
        "url": "https://www.kaggle.com/learn/pandas",
        "time": "~4 hours",
        "type": "Hands-on course",
    },
    "power bi": {
        "title": "Microsoft Learn: Training for Power BI",
        "provider": "Microsoft Learn",
        "url": "https://learn.microsoft.com/en-us/training/powerplatform/power-bi",
        "time": "Self-paced",
        "type": "Official learning path",
    },
    "statistics": {
        "title": "Google Advanced Data Analytics Professional Certificate",
        "provider": "Coursera",
        "url": "https://www.coursera.org/professional-certificates/google-advanced-data-analytics",
        "time": "Self-paced",
        "type": "Professional certificate",
    },
    "data analysis": {
        "title": "Google Data Analytics Professional Certificate",
        "provider": "Coursera",
        "url": "https://www.coursera.org/professional-certificates/google-data-analytics",
        "time": "Self-paced",
        "type": "Professional certificate",
    },
    "data visualization": {
        "title": "Microsoft Learn: Prepare and Visualize Data with Power BI",
        "provider": "Microsoft Learn",
        "url": "https://learn.microsoft.com/en-us/training/paths/prepare-visualize-data-power-bi/",
        "time": "Self-paced",
        "type": "Official learning path",
    },
    "machine learning": {
        "title": "Kaggle Learn: Intro to Machine Learning",
        "provider": "Kaggle",
        "url": "https://www.kaggle.com/learn/intro-to-machine-learning",
        "time": "~3 hours",
        "type": "Hands-on course",
    },
    "excel": {
        "title": "Google Data Analytics Certificate",
        "provider": "Coursera",
        "url": "https://www.coursera.org/professional-certificates/google-data-analytics",
        "time": "Self-paced",
        "type": "Certificate path",
    },
}

# =====================================================
# HELPERS
# =====================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+# ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_read_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    try:
        raw = uploaded_file.getvalue()
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def clean_user_skills(skills: list[str]) -> list[str]:
    return sorted({normalize_text(s) for s in skills if normalize_text(s)})


def extract_skills_from_text(text: str) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []

    found = []
    for skill, keywords in SKILL_DICT.items():
        for keyword in keywords:
            kw = normalize_text(keyword)
            if not kw:
                continue
            if f" {kw} " in f" {cleaned} ":
                found.append(skill)
                break
    return sorted(set(found))


def filtered_skills(skills: list[str]) -> list[str]:
    return [s for s in skills if s not in NOISY_SKILLS]


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def keyword_relevance(query: str, title: str, description: str, skills: list[str]) -> float:
    query = normalize_text(query)
    if not query:
        return 0.0

    title_text = normalize_text(title)
    desc_text = normalize_text(description)
    skills_text = normalize_text(" ".join(skills))

    score = 0.0
    query_terms = [q for q in query.split() if len(q) > 1]

    if query in title_text:
        score += 50
    if query in skills_text:
        score += 35
    if query in desc_text:
        score += 20

    for term in query_terms:
        if term in title_text:
            score += 14
        if term in skills_text:
            score += 9
        if term in desc_text:
            score += 4

    title_sim = similarity(query, title)
    score += title_sim * 20
    return score


def calculate_skill_match(user_skills: list[str], job_skills: list[str]) -> tuple[int, list[str], list[str]]:
    user_core = set(filtered_skills(user_skills))
    job_core = set(filtered_skills(job_skills))

    if not user_core or not job_core:
        return 0, [], sorted(job_core)

    matched = sorted(user_core & job_core)
    missing = sorted(job_core - user_core)
    if not matched or len(matched) == 0:
        return 0, [], missing

    coverage = len(matched) / len(job_core)
    closeness = len(matched) / len(user_core)
    score = round((coverage * 0.75 + closeness * 0.25) * 100)
    return min(score, 100), matched, missing


def build_user_profile_text() -> str:
    parts = [
        st.session_state.get("target_role", ""),
        st.session_state.get("interests", ""),
        st.session_state.get("projects", ""),
        st.session_state.get("manual_skills", ""),
        st.session_state.get("cv_text", ""),
        st.session_state.get("transcript_text", ""),
        " ".join(st.session_state.get("final_user_skills", [])),
    ]
    return normalize_text(" ".join([p for p in parts if p]))


def build_job_text(row: pd.Series) -> str:
    parts = [
        row.get("title", ""),
        row.get("description", ""),
        " ".join(row.get("skills", [])),
        row.get("source", ""),
        row.get("location", ""),
    ]
    return normalize_text(" ".join([str(p) for p in parts if str(p).strip()]))


def compute_tfidf_scores(user_text: str, jobs_df: pd.DataFrame) -> list[float]:
    if not user_text.strip() or jobs_df.empty:
        return [0.0] * len(jobs_df)

    job_texts = [build_job_text(row) for _, row in jobs_df.iterrows()]
    corpus = [user_text] + job_texts

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus)
        user_vec = tfidf_matrix[0:1]
        job_vecs = tfidf_matrix[1:]
        sims = cosine_similarity(user_vec, job_vecs).flatten()
        return [round(float(score) * 100, 2) for score in sims]
    except Exception:
        return [0.0] * len(jobs_df)


def recommend_jobs(user_skills: list[str], data: pd.DataFrame, preferred_location: str = "All locations", search_query: str = "") -> pd.DataFrame:
    temp_df = data.copy()

    if preferred_location and preferred_location != "All locations":
        temp_df = temp_df[
            temp_df["location"].astype(str).str.lower().str.contains(preferred_location.lower(), na=False)
        ]

    if temp_df.empty:
        return pd.DataFrame()

    user_text = build_user_profile_text()
    tfidf_scores = compute_tfidf_scores(user_text, temp_df)

    rows = []
    for idx, (_, row) in enumerate(temp_df.iterrows()):
        base_score, aligned, missing = calculate_skill_match(user_skills, row["skills"])

        if base_score <= 0:
            continue

        # remove weak matches caused by generic skills
        if len(aligned) == 0:
            continue

        if len(aligned) < 2 and "analyst" in st.session_state.get("target_role", "").lower():
            base_score = round(base_score * 0.7)

        keyword_score = keyword_relevance(search_query, row["title"], row["description"], row["skills"])
        tfidf_score = tfidf_scores[idx] if idx < len(tfidf_scores) else 0.0

        final_score = round(min(100, base_score * 0.75 + tfidf_score * 0.20 + keyword_score * 0.05))

        title_lower = str(row["title"]).lower()
        target_role_lower = st.session_state.get("target_role", "").lower()

        junior_keywords = ["junior", "intern", "analyst", "associate", "entry"]
        senior_keywords = ["chief", "director", "head", "vp", "vice president", "lead", "senior"]
        irrelevant_keywords = ["reception", "secretary", "executive assistant", "assistant", "administrator", "admin"]
        preferred_role_keywords = ["data", "analyst", "business intelligence", "bi", "reporting", "dashboard"]

        if any(word in title_lower for word in junior_keywords):
            final_score += 10

        if any(word in title_lower for word in senior_keywords):
            final_score -= 20

        if any(word in title_lower for word in irrelevant_keywords):
            final_score -= 20

        if "analyst" in target_role_lower:
            if any(word in title_lower for word in preferred_role_keywords):
                final_score += 12
            else:
                final_score -= 10

        final_score = max(0, min(100, final_score))

        if final_score < 10:
            continue

        rows.append(
            {
                "title": row["title"],
                "source": row["source"],
                "location": row["location"],
                "url": row["url"],
                "description": row["description"],
                "job_skills": row["skills"],
                "aligned_skills": aligned,
                "missing_skills": missing,
                "skill_match": base_score,
                "ml_score": round(tfidf_score, 2),
                "relevance_score": round(keyword_score, 1),
                "match_percent": final_score,
            }
        )

    results = pd.DataFrame(rows)

    if results.empty:
        # fallback: show top jobs based only on ML similarity
        temp_df["fallback_score"] = tfidf_scores
        fallback = temp_df.sort_values(by="fallback_score", ascending=False).head(5)

        rows = []
        for _, row in fallback.iterrows():
            rows.append({
                "title": row["title"],
                "source": row["source"],
                "location": row["location"],
                "url": row["url"],
                "description": row["description"],
                "job_skills": row["skills"],
                "aligned_skills": [],
                "missing_skills": row["skills"],
                "skill_match": 0,
                "ml_score": row["fallback_score"],
                "relevance_score": 0,
                "match_percent": round(row["fallback_score"])
             })

    results = pd.DataFrame(rows)
    

    results = results.sort_values(
        by=["match_percent", "skill_match", "ml_score", "relevance_score", "title"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    return results


def learning_resource_for_skill(skill: str) -> dict:
    if skill in LEARNING_RESOURCES:
        return LEARNING_RESOURCES[skill]

    query = quote(f"{skill} course certification")
    return {
        "title": f"Search a course for {skill.title()}",
        "provider": "Search",
        "url": f"https://www.google.com/search?q={query}",
        "time": "Varies",
        "type": "Suggested search",
    }


def build_real_plan(selected_row: pd.Series | None) -> list[dict]:
    if selected_row is None:
        return []

    missing = list(selected_row.get("missing_skills", []))
    aligned = list(selected_row.get("aligned_skills", []))
    title = selected_row.get("title", "target role")

    if not missing:
        return [
            {
                "title": "Tailor your CV to this role",
                "detail": f"Update your CV so it highlights the skills already aligned with {title}.",
                "time": "Est. 1 hour",
                "url": "",
                "provider": "Profile optimization",
            },
            {
                "title": "Prepare 2 portfolio examples",
                "detail": "Choose two projects or assignments that prove your current strengths.",
                "time": "Est. 2 hours",
                "url": "",
                "provider": "Portfolio task",
            },
            {
                "title": "Practice role-specific interview questions",
                "detail": f"Prepare examples using your matched skills: {', '.join(aligned[:4]) if aligned else 'your current strengths'}.",
                "time": "Est. 2 hours",
                "url": "",
                "provider": "Interview prep",
            },
            {
                "title": "Apply to similar jobs this week",
                "detail": "Start with the strongest matches and track your applications.",
                "time": "Est. 1 hour",
                "url": "",
                "provider": "Action step",
            },
        ]

    top_missing = missing[:2]
    plan = []

    for skill in top_missing:
        resource = learning_resource_for_skill(skill)
        plan.append(
            {
                "title": f"Learn {skill.title()}",
                "detail": f"{resource['title']} — {resource['type']} by {resource['provider']}.",
                "time": resource["time"],
                "url": resource["url"],
                "provider": resource["provider"],
            }
        )

    combined_skills = " and ".join([s.title() for s in top_missing])
    plan.append(
        {
            "title": "Complete one guided practice task",
            "detail": f"Use a small dataset or sample business case to practice {combined_skills} in a real scenario.",
            "time": "Est. 3 hours",
            "url": "https://www.kaggle.com/datasets",
            "provider": "Kaggle Datasets",
        }
    )
    plan.append(
        {
            "title": "Build one portfolio-ready mini project",
            "detail": f"Create a small project that demonstrates {combined_skills}, then add it to GitHub or your CV.",
            "time": "Est. 4–6 hours",
            "url": "",
            "provider": "Portfolio task",
        }
    )

    return plan[:4]


def combine_user_skills() -> list[str]:
    sources = [
        st.session_state.get("manual_skills", ""),
        st.session_state.get("interests", ""),
        st.session_state.get("target_role", ""),
        st.session_state.get("cv_text", ""),
        st.session_state.get("transcript_text", ""),
    ]
    extracted = []
    for source in sources:
        extracted.extend(extract_skills_from_text(source))

    manual = clean_user_skills(st.session_state.get("manual_skills", "").split(","))
    final_skills = sorted(set(extracted + manual))
    return final_skills


def initials(name: str) -> str:
    parts = [p for p in name.split() if p.strip()]
    if not parts:
        return "SM"
    return "".join(p[0] for p in parts[:2]).upper()


def rerun_matches():
    skills = combine_user_skills()
    st.session_state["final_user_skills"] = skills
    st.session_state["results"] = recommend_jobs(
        user_skills=skills,
        data=JOBS_DF,
        preferred_location=st.session_state.get("preferred_location", "All locations"),
        search_query=st.session_state.get("search_query", ""),
    )
    st.session_state["selected_result_idx"] = 0


def to_download_csv(df: pd.DataFrame) -> bytes:
    if df.empty:
        return pd.DataFrame(columns=["title", "source", "location", "match_percent", "aligned_skills", "missing_skills", "url"]).to_csv(index=False).encode("utf-8")
    export_df = df[["title", "source", "location", "match_percent", "aligned_skills", "missing_skills", "url"]].copy()
    return export_df.to_csv(index=False).encode("utf-8")

# =====================================================
# SESSION STATE
# =====================================================

def init_state():
    defaults = {
        "page": "landing",
        "full_name": "",
        "email": "",
        "password": "",
        "major": "",
        "year": "",
        "city": "",
        "gpa": "",
        "interests": "",
        "target_role": "",
        "manual_skills": "",
        "projects": "",
        "cv_text": "",
        "transcript_text": "",
        "final_user_skills": [],
        "results": pd.DataFrame(),
        "selected_result_idx": 0,
        "search_query": "",
        "preferred_location": "All locations",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

# =====================================================
# STYLES
# =====================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(59,130,246,0.12), transparent 30%),
            radial-gradient(circle at top left, rgba(20,184,166,0.10), transparent 30%),
            linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
    }

    .hero {
        background:
            radial-gradient(circle at top right, rgba(59,130,246,0.25), transparent 35%),
            radial-gradient(circle at bottom left, rgba(20,184,166,0.22), transparent 40%),
            linear-gradient(135deg, #0b1f33 0%, #0f766e 45%, #14b8a6 75%, #3b82f6 100%);
        color: white;
        border-radius: 30px;
        padding: 48px 44px;
        box-shadow: 0 22px 60px rgba(2, 8, 23, 0.45);
        margin-top: 30px;
        margin-bottom: 30px;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 999px;
        padding: 8px 14px;
        font-size: 0.9rem;
        margin-bottom: 18px;
        font-weight: 600;
    }

    .hero-title {
        font-size: 3.8rem;
        font-weight: 800;
        line-height: 1.04;
        margin-bottom: 12px;
    }

    .hero-sub {
        max-width: 720px;
        font-size: 1.08rem;
        line-height: 1.7;
        color: rgba(255,255,255,0.82);
        margin-top: 14px;
    }

    .glass {
        background: rgba(255,255,255,0.82);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.55);
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        padding: 22px;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 8px;
    }

    .muted {
        color: #6b7280;
        font-size: 0.96rem;
        line-height: 1.6;
    }

    .dashboard-shell {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 28px;
        overflow: hidden;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        margin-bottom: 18px;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 22px 28px;
        border-bottom: 1px solid #e5e7eb;
        background: #fafafa;
    }

    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .logo-circle {
        width: 58px;
        height: 58px;
        border-radius: 50%;
        background: linear-gradient(135deg, #14b8a6 0%, #3b82f6 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        font-weight: 800;
    }

    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 3px;
    }

    .brand-sub {
        color: #6b7280;
        font-size: 0.96rem;
    }

    .main-card {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 22px;
        height: 100%;
    }

    .profile-top {
        display: flex;
        gap: 14px;
        align-items: flex-start;
        margin-bottom: 18px;
    }

    .avatar {
        width: 82px;
        height: 82px;
        border-radius: 50%;
        background: #e5e7eb;
        color: #111827;
        font-size: 2rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    .profile-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1f2937;
    }

    .profile-meta {
        color: #6b7280;
        font-size: 0.96rem;
        line-height: 1.45;
    }

    .chip {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 10px;
        margin: 0 8px 8px 0;
        font-size: 0.9rem;
        border: 1px solid transparent;
    }

    .chip.good {
        background: #ecfdf3;
        border-color: #b7ebc6;
        color: #2f855a;
    }

    .chip.bad {
        background: #fef3f2;
        border-color: #f5c2c0;
        color: #c0392b;
    }

    .job-shell {
        background: #ffffff;
        border: 1px solid #dbe3ea;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }

    .job-title {
        font-size: 1.03rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 2px;
    }

    .job-source {
        color: #6b7280;
        margin-bottom: 10px;
    }

    .score-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        gap: 12px;
    }

    .score-num {
        font-size: 1.1rem;
        font-weight: 800;
        color: #0f766e;
    }

    .progress-wrap {
        width: 100%;
        height: 12px;
        background: #edf2f7;
        border-radius: 999px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #14b8a6 0%, #0ea5e9 100%);
        border-radius: 999px;
    }

    .plan-step {
        display: flex;
        gap: 14px;
        align-items: flex-start;
        margin-bottom: 16px;
    }

    .step-badge {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #0f766e;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        flex-shrink: 0;
    }

    .step-title {
        font-weight: 700;
        color: #111827;
        margin-bottom: 3px;
    }

    .step-time {
        color: #98a2b3;
        font-size: 0.92rem;
        margin-bottom: 3px;
    }

    .step-detail {
        color: #475569;
        font-size: 0.94rem;
        line-height: 1.5;
    }

    .tiny-note {
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.6;
        margin-top: 10px;
    }

    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        border-radius: 14px;
        min-height: 48px;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #14b8a6);
        color: white;
        border: none;
        box-shadow: 0 6px 16px rgba(20,184,166,0.25);
        transition: all 0.2s ease;
    }

    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(20,184,166,0.35);
    }
}

    .metric {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(15,23,42,0.04);
    }

    .metric-num {
        font-size: 1.8rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 4px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.94rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# PAGE 1: LANDING
# =====================================================

# =====================================================
# PAGE 1: LANDING
# =====================================================

def landing_page():
    hero_left, hero_right = st.columns([1.3, 0.7], gap="large")

    with hero_left:
        st.markdown(
            """
            <div class="hero">
                <div class="hero-badge">AI-powered career navigation for university students</div>
                <div class="hero-title">Find your perfect career match — powered by AI</div>
                <div class="hero-sub">
                    Upload your CV or transcript, uncover your strongest skills, and explore career opportunities with personalized skill-gap insights and action plans.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        if st.button("🚀 Start Matching", key="landing_start"):
            st.session_state["page"] = "profile"
            st.rerun()

    with hero_right:
        st.markdown(
            """
            <div class="hero" style="display:flex; align-items:center; justify-content:center; min-height:360px; padding:40px;">
                <div style="text-align:center;">
                    <div style="font-size:5rem; line-height:1;">🧠</div>
                    <div style="font-size:1.2rem; font-weight:700; margin-top:12px;">Smart Career Matching</div>
                    <div style="color:rgba(255,255,255,0.82); margin-top:8px; line-height:1.6;">
                        AI + NLP + ML working together to map skills, rank jobs, and recommend next steps.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            f'<div class="metric"><div class="metric-num">{len(JOBS_DF)}</div><div class="metric-label">UAE job postings loaded</div></div>',
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            f'<div class="metric"><div class="metric-num">{JOBS_DF["source"].nunique()}</div><div class="metric-label">Job sources combined</div></div>',
            unsafe_allow_html=True
        )

    with m3:
        all_skills = [s for row in JOBS_DF["skills"] for s in row]
        st.markdown(
            f'<div class="metric"><div class="metric-num">{len(set(all_skills))}</div><div class="metric-label">Distinct extracted skills</div></div>',
            unsafe_allow_html=True
        )

    st.write("")
    c1, c2 = st.columns([1.2, 0.8], gap="large")

    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="muted">
            1. Create your profile.<br>
            2. Upload your CV and transcript or add skills manually.<br>
            3. Get ranked matches based on skills, role interest, and job content.<br>
            4. Review matched skills, missing skills, and a real learning plan with online resources.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

        st.markdown(
        """
        <div style="
        padding:20px;
        margin-top:10px;
        background: linear-gradient(135deg, #0f766e, #3b82f6);
        border-radius:18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        color:white;
        ">
        <div style="font-weight:700; font-size:1.1rem; margin-bottom:10px;">
        Example Output
        </div>

        <div style="font-size:0.95rem; line-height:1.7;">
        <b>Junior Data Analyst</b> —
        <span style="color:#a7f3d0; font-weight:700;">86% match</span><br><br>

        <b>Matched skills:</b> Python, SQL, Excel<br>
        <b>Missing skills:</b> Pandas, Statistics<br><br>

        <b>Next step:</b> Complete a short Pandas course and build one mini project.
        </div>
        </div>""",
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)
    
    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Try it</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="muted">Start with a profile, then upload your information and generate your personalized dashboard.</div>',
            unsafe_allow_html=True
        )
        st.write("")
        if st.button("Get Started", type="primary", use_container_width=True, key="landing_get_started"):
            st.session_state["page"] = "profile"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# PAGE 2: PROFILE / CREATE ACCOUNT
# =====================================================

def profile_page():
    st.markdown('<div class="section-title" style="font-size:1.6rem;">Create your profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">This is a profile setup step for the demo. It is not a persistent account database yet.</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.session_state["full_name"] = st.text_input("Full name", value=st.session_state["full_name"], placeholder="e.g. Alyazia Alameri")
        st.session_state["email"] = st.text_input("Email", value=st.session_state["email"], placeholder="e.g. name@email.com")
        st.session_state["password"] = st.text_input("Create password", value=st.session_state["password"], type="password", placeholder="For demo only")
        st.session_state["major"] = st.text_input("Major / concentration", value=st.session_state["major"], placeholder="e.g. Computational Systems")
    with c2:
        st.session_state["year"] = st.text_input("Study level", value=st.session_state["year"], placeholder="e.g. Final Year")
        st.session_state["city"] = st.text_input("City", value=st.session_state["city"], placeholder="e.g. Abu Dhabi")
        st.session_state["gpa"] = st.text_input("GPA (optional)", value=st.session_state["gpa"], placeholder="e.g. 3.5")
        st.session_state["interests"] = st.text_input("Career interests", value=st.session_state["interests"], placeholder="e.g. AI, data analytics, cybersecurity")

    st.session_state["target_role"] = st.text_input("Target role or field", value=st.session_state["target_role"], placeholder="e.g. Junior Data Analyst")
    st.session_state["projects"] = st.text_area("Projects / coursework / experience", value=st.session_state["projects"], height=100, placeholder="Tell us about your projects, coursework, internships, or achievements")

    b1, b2 = st.columns([0.18, 0.82])
    with b1:
        if st.button("Back", use_container_width=True):
            st.session_state["page"] = "landing"
            st.rerun()
    with b2:
        if st.button("Continue to Uploads", type="primary", use_container_width=True):
            if not st.session_state["full_name"] or not st.session_state["major"]:
                st.warning("Please enter at least your name and major before continuing.")
            else:
                st.session_state["page"] = "try"
                st.rerun()

# =====================================================
# PAGE 3: TRY IT / UPLOAD
# =====================================================

def try_page():
    st.markdown(f'<div class="section-title" style="font-size:1.6rem;">Hello, {st.session_state.get("full_name") or "there"} 👋</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Upload your files, add skills manually if needed, and generate your personalized match dashboard.</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        cv_file = st.file_uploader("Upload your CV", type=["txt", "pdf", "docx"], key="cv_upload_real")
        transcript_file = st.file_uploader("Upload your transcript", type=["txt", "pdf", "docx"], key="transcript_upload_real")

        if cv_file is not None:
            st.session_state["cv_text"] = safe_read_uploaded_file(cv_file)
        if transcript_file is not None:
            st.session_state["transcript_text"] = safe_read_uploaded_file(transcript_file)

    with c2:
        st.session_state["manual_skills"] = st.text_area(
            "Add skills manually",
            value=st.session_state["manual_skills"],
            height=120,
            placeholder="e.g. python, sql, power bi, data visualization",
        )
        location_options = ["All locations"] + sorted([str(x) for x in JOBS_DF["location"].dropna().unique() if str(x).strip()])
        current = st.session_state["preferred_location"]
        current_idx = location_options.index(current) if current in location_options else 0
        st.session_state["preferred_location"] = st.selectbox("Preferred job location", location_options, index=current_idx)
        st.session_state["search_query"] = st.text_input(
            "Search focus (optional)",
            value=st.session_state["search_query"],
            placeholder="e.g. data analyst, cybersecurity, power bi",
        )

    extracted_preview = combine_user_skills()
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Detected profile skills</div>', unsafe_allow_html=True)
    if extracted_preview:
        for skill in extracted_preview:
            st.markdown(f'<span class="chip good">{skill.title()}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="muted">No skills detected yet. Upload a file or add skills manually.</div>', unsafe_allow_html=True)
    st.markdown('<div class="tiny-note">Note: PDF and DOCX support is basic in this version. Plain text files work best right now unless we add dedicated parsers next.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    b1, b2 = st.columns([0.18, 0.82])
    with b1:
        if st.button("Back", use_container_width=True):
            st.session_state["page"] = "profile"
            st.rerun()
    with b2:
        if st.button("Generate My Match", type="primary", use_container_width=True):
            rerun_matches()
            if st.session_state["results"].empty:
                st.warning("No strong matches were found with the current profile. Try adding more specific skills or adjusting your search focus.")
            else:
                st.session_state["page"] = "dashboard"
                st.rerun()

# =====================================================
# PAGE 4: DASHBOARD
# =====================================================

def dashboard_page():
    results = st.session_state.get("results", pd.DataFrame())
    selected_row = None
    if not results.empty:
        idx = min(st.session_state.get("selected_result_idx", 0), len(results) - 1)
        selected_row = results.iloc[idx]

    topbar_left, topbar_right = st.columns([0.76, 0.24])
    with topbar_left:
        st.markdown(
            f"""
            <div class="dashboard-shell">
                <div class="topbar">
                    <div class="brand-wrap">
                        <div class="logo-circle">{initials(st.session_state.get('full_name', 'SM'))}</div>
                        <div>
                            <div class="brand-title">SkillMatch AI — Personalized Dashboard</div>
                            <div class="brand-sub">Hello, {st.session_state.get('full_name') or 'User'} • Upload → Match → Plan</div>
                        </div>
                    </div>
                    <div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with topbar_right:
        st.download_button(
            "Export CSV",
            data=to_download_csv(results),
            file_name="skillmatch_results.csv",
            use_container_width=True,
        )
        if st.button("New Search", type="primary", use_container_width=True):
            st.session_state["page"] = "try"
            st.rerun()

    with st.expander("Update profile or search settings", expanded=False):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.session_state["full_name"] = st.text_input("Full name", value=st.session_state["full_name"], key="dash_name")
            st.session_state["major"] = st.text_input("Major", value=st.session_state["major"], key="dash_major")
            st.session_state["year"] = st.text_input("Study level", value=st.session_state["year"], key="dash_year")
        with c2:
            st.session_state["city"] = st.text_input("City", value=st.session_state["city"], key="dash_city")
            st.session_state["gpa"] = st.text_input("GPA", value=st.session_state["gpa"], key="dash_gpa")
            st.session_state["target_role"] = st.text_input("Target role", value=st.session_state["target_role"], key="dash_target")
        with c3:
            st.session_state["manual_skills"] = st.text_area("Manual skills", value=st.session_state["manual_skills"], height=90, key="dash_skills")
            location_options = ["All locations"] + sorted([str(x) for x in JOBS_DF["location"].dropna().unique() if str(x).strip()])
            current = st.session_state["preferred_location"]
            current_idx = location_options.index(current) if current in location_options else 0
            st.session_state["preferred_location"] = st.selectbox("Preferred location", location_options, index=current_idx, key="dash_location")
            st.session_state["search_query"] = st.text_input("Search focus", value=st.session_state["search_query"], key="dash_search")
        if st.button("Refresh Matches", use_container_width=True):
            rerun_matches()
            st.rerun()

    left_col, mid_col, right_col = st.columns([1.0, 1.7, 1.3], gap="large")

    with left_col:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="profile-top">
                <div class="avatar">{initials(st.session_state.get('full_name', 'SM'))}</div>
                <div>
                    <div class="profile-name">{st.session_state.get('full_name') or 'User'}</div>
                    <div class="profile-meta">{st.session_state.get('major') or 'No major added'} • {st.session_state.get('year') or 'Study level not set'}<br>{st.session_state.get('city') or 'Location not set'}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-title">Key skills</div>', unsafe_allow_html=True)
        if st.session_state.get("final_user_skills"):
            for skill in st.session_state["final_user_skills"][:10]:
                st.markdown(f'<span class="chip good">{skill.title()}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="muted">No skills detected yet.</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="muted" style="margin-top:12px;"><strong>GPA:</strong> {st.session_state.get("gpa") or "Not provided"}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="muted" style="margin-top:8px;"><strong>Target role:</strong> {st.session_state.get("target_role") or "Not provided"}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="muted" style="margin-top:8px;"><strong>Projects:</strong> {st.session_state.get("projects") or "No projects added yet"}</div>', unsafe_allow_html=True)
        st.write("")
        st.button("Edit profile", use_container_width=True)
        st.button("Run skill audit", use_container_width=True)
        st.button("Save report", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with mid_col:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-title">Top job matches</div><div class="muted">{len(results)} results • hybrid ranking using rule-based matching + TF-IDF ML similarity</div>',
            unsafe_allow_html=True,
        )

        if results.empty:
            st.info("No results available yet. Go back and generate your match.")
        else:
            for i, row in results.head(8).iterrows():
                st.markdown(
                    f"""
                    <div class="job-shell">
                        <div class="score-row">
                            <div>
                                <div class="job-title">{row['title']}</div>
                                <div class="job-source">{row['source']} • {row['location']}</div>
                            </div>
                            <div style="text-align:right;">
                                <div class="score-num">{row['match_percent']}%</div>
                                <div class="muted">Match score</div>
                            </div>
                        </div>
                        <div class="muted" style="margin-bottom:10px;"><strong>Matched:</strong> {', '.join(row['aligned_skills']) if row['aligned_skills'] else '—'}</div>
                        <div class="muted" style="margin-bottom:10px;"><strong>ML score:</strong> {row['ml_score']} • <strong>Skill match:</strong> {row['skill_match']}</div>
                        <div class="progress-wrap"><div class="progress-fill" style="width:{row['match_percent']}%;"></div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns([0.76, 0.24])
                with c1:
                    pass
                with c2:
                    if st.button("View", key=f"view_job_{i}", use_container_width=True):
                        st.session_state["selected_result_idx"] = i
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        if selected_row is None:
            st.info("Select a job card to see the skill-gap analysis and learning plan.")
        else:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; gap:12px; margin-bottom:14px;">
                    <div class="section-title" style="margin:0;">Skill-gap ({selected_row['title']})</div>
                    <div class="muted">Selected: {selected_row['source']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-title" style="font-size:0.95rem;">Matched skills</div>', unsafe_allow_html=True)
            if selected_row["aligned_skills"]:
                for skill in selected_row["aligned_skills"]:
                    st.markdown(f'<span class="chip good">{skill.title()}</span>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="muted">No strong overlaps detected yet.</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title" style="font-size:0.95rem; margin-top:14px;">Missing skills</div>', unsafe_allow_html=True)
            if selected_row["missing_skills"]:
                for skill in selected_row["missing_skills"]:
                    st.markdown(f'<span class="chip bad">{skill.title()}</span>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="muted">No major gaps identified for this role.</div>', unsafe_allow_html=True)

            st.markdown('<hr style="border:none;border-top:1px solid #e5e7eb;margin:18px 0;">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">4-step learning plan</div>', unsafe_allow_html=True)
            plan = build_real_plan(selected_row)
            for idx, step in enumerate(plan, start=1):
                st.markdown(
                    f"""
                    <div class="plan-step">
                        <div class="step-badge">{idx}</div>
                        <div>
                            <div class="step-title">{step['title']}</div>
                            <div class="step-time">{step['time']}</div>
                            <div class="step-detail">{step['detail']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if step.get("url"):
                    st.link_button(f"Open resource {idx}", step["url"], use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.button("Start first step", use_container_width=True, type="primary")
            with c2:
                st.button("Save plan", use_container_width=True)
            st.markdown('<div class="tiny-note">These plan steps are tied to real learning resources where available. Job ranking now combines explicit skill overlap with TF-IDF machine learning similarity.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c_back, c_home = st.columns([0.16, 0.84])
    with c_back:
        if st.button("Back", use_container_width=True):
            st.session_state["page"] = "try"
            st.rerun()
    with c_home:
        if st.button("Home", use_container_width=True):
            st.session_state["page"] = "landing"
            st.rerun()

# =====================================================
# ROUTER
# =====================================================

page = st.session_state.get("page", "landing")

if page == "landing":
    landing_page()
elif page == "profile":
    profile_page()
elif page == "try":
    try_page()
else:
    dashboard_page()
