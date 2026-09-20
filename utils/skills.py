"""
Skill vocabulary and alias map for JobMatch AI.

The dataset uses a CLOSED vocabulary of 73 canonical skill names, shared by
both resumes and job postings. That makes dictionary-based skill extraction
the right tool here: it is exact, fast and fully explainable.

Real resumes, however, do not use the canonical spelling. A candidate writes
"Postgres", "object oriented programming" or "split testing".
ALIASES maps those surface forms back to a canonical skill so that free text
(for example a PDF uploaded in the Streamlit app) can be handled too.

Every alias below is a common real-world synonym or abbreviation of the
canonical skill. No alias invents a skill that does not exist in the dataset.
"""

# --- 1. Canonical vocabulary (extracted from the dataset, not invented) -----
SKILL_VOCAB = [
    "A/B Testing", "Account Management", "Accounting", "Agile", "Analytics",
    "Asana", "Budgeting", "CI/CD", "CRM", "Cash Flow", "Closing",
    "Communication", "Content Marketing", "Conversion Optimization",
    "Copywriting", "Cross-functional Coordination", "Customer Satisfaction",
    "Data Visualization", "Databases", "Discovery Calls", "Docker",
    "Documentation", "ETL", "Email Marketing", "Escalations", "Excel",
    "Financial Modeling", "Forecasting", "Git", "Google Ads", "Intercom",
    "Java", "JavaScript", "Jira", "KPIs", "Landing Pages", "Lead Generation",
    "Marketing Analytics", "Meta Ads", "Negotiation", "OOP",
    "Outbound Outreach", "PRD", "Pandas", "Pipeline Management", "Power BI",
    "Prioritization", "Process Improvement", "Product Strategy",
    "Project Planning", "Prospecting", "Python", "REST APIs", "Reporting",
    "Risk Management", "Roadmap", "Root Cause Analysis", "SEO", "SLA", "SQL",
    "Scrum", "Stakeholder Communication", "Stakeholder Management",
    "Statistics", "Tableau", "Ticketing", "Timeline Management",
    "Troubleshooting", "Unit Testing", "User Research", "Valuation",
    "Variance Analysis", "Zendesk",
]

# --- 2. Alias -> canonical skill -------------------------------------------
ALIASES = {
    "A/B Testing": ["ab testing", "a b testing", "split testing", "split tests",
                    "experimentation", "multivariate testing"],
    "Account Management": ["account manager", "key accounts", "client management",
                           "relationship management"],
    "Accounting": ["bookkeeping", "general ledger", "accounts payable",
                   "accounts receivable", "gaap"],
    "Agile": ["agile methodology", "agile development", "kanban", "sprints"],
    "Analytics": ["data analytics", "google analytics", "ga4", "web analytics"],
    "Asana": ["asana boards"],
    "Budgeting": ["budget planning", "budget management", "cost planning"],
    "CI/CD": ["ci cd", "continuous integration", "continuous delivery",
              "continuous deployment", "github actions", "jenkins",
              "build pipelines"],
    "CRM": ["customer relationship management", "salesforce", "hubspot",
            "pipedrive"],
    "Cash Flow": ["cashflow", "cash flow management", "liquidity",
                  "working capital"],
    "Closing": ["closing deals", "deal closing", "sales closing", "win rate"],
    "Communication": ["verbal communication", "written communication",
                      "communication skills", "presentation skills"],
    "Content Marketing": ["content strategy", "blogging", "content creation"],
    "Conversion Optimization": ["cro", "conversion rate optimization",
                                "funnel optimization"],
    "Copywriting": ["copy writing", "ad copy", "content writing"],
    "Cross-functional Coordination": ["cross functional",
                                      "cross-functional teams",
                                      "interdepartmental coordination"],
    "Customer Satisfaction": ["csat", "nps", "customer experience",
                              "customer happiness"],
    "Data Visualization": ["dataviz", "dashboards", "charts", "matplotlib",
                           "visualisation", "visualization"],
    "Databases": ["database", "postgresql", "postgres", "mysql", "mongodb",
                  "relational databases", "rdbms"],
    "Discovery Calls": ["discovery call", "needs analysis",
                        "qualification calls"],
    "Docker": ["containers", "containerization", "docker compose"],
    "Documentation": ["technical writing", "documenting", "runbooks",
                      "confluence"],
    "ETL": ["etl pipelines", "elt", "data pipelines", "data ingestion",
            "extract transform load"],
    "Email Marketing": ["email campaigns", "newsletters", "mailchimp",
                        "drip campaigns"],
    "Escalations": ["escalation management", "tier 2 support",
                    "tier 3 support"],
    "Excel": ["microsoft excel", "spreadsheets", "google sheets",
              "pivot tables", "vlookup"],
    "Financial Modeling": ["financial models", "three statement model",
                           "dcf model", "financial modelling"],
    "Forecasting": ["demand forecasting", "revenue forecasting", "projections"],
    "Git": ["github", "gitlab", "version control", "source control"],
    "Google Ads": ["adwords", "google adwords", "sem", "paid search", "ppc"],
    "Intercom": ["intercom chat"],
    "Java": ["java 8", "java 11", "spring boot", "jvm"],
    "JavaScript": ["js", "es6", "typescript", "node js", "nodejs", "react"],
    "Jira": ["jira tickets", "atlassian jira"],
    "KPIs": ["kpi", "key performance indicators", "okrs", "metrics tracking"],
    "Landing Pages": ["landing page", "lp optimization", "webflow pages"],
    "Lead Generation": ["lead gen", "demand generation", "demand gen",
                        "inbound leads"],
    "Marketing Analytics": ["campaign analytics", "attribution",
                            "roas analysis"],
    "Meta Ads": ["facebook ads", "instagram ads", "meta advertising",
                 "paid social"],
    "Negotiation": ["contract negotiation", "negotiating", "deal negotiation"],
    "OOP": ["object oriented programming", "object-oriented programming",
            "object oriented design", "oop principles"],
    "Outbound Outreach": ["outbound", "cold outreach", "cold calling",
                          "cold emailing", "sdr outreach"],
    "PRD": ["product requirements document", "product requirement documents",
            "product specs", "product specifications"],
    "Pandas": ["pandas dataframe", "python pandas", "numpy"],
    "Pipeline Management": ["sales pipeline", "pipeline hygiene",
                            "opportunity management"],
    "Power BI": ["powerbi", "microsoft power bi", "power bi dashboards", "dax"],
    "Prioritization": ["prioritisation", "backlog prioritization",
                       "roadmap prioritization", "moscow"],
    "Process Improvement": ["process optimization", "continuous improvement",
                            "lean", "six sigma", "workflow improvement"],
    "Product Strategy": ["product vision", "go to market strategy",
                         "product positioning"],
    "Project Planning": ["project management", "work breakdown structure",
                         "gantt", "project scheduling"],
    "Prospecting": ["prospect research", "lead sourcing", "list building"],
    "Python": ["python 3", "python scripting"],
    "REST APIs": ["rest api", "restful api", "restful apis", "web services",
                  "api development", "http apis"],
    "Reporting": ["reports", "management reporting", "weekly reporting",
                  "business reporting"],
    "Risk Management": ["risk assessment", "risk mitigation", "risk analysis"],
    "Roadmap": ["product roadmap", "roadmapping", "roadmap planning"],
    "Root Cause Analysis": ["rca", "root cause", "5 whys", "problem analysis"],
    "SEO": ["search engine optimization", "search engine optimisation",
            "organic search", "on page seo"],
    "SLA": ["service level agreement", "service level agreements",
            "sla compliance", "response time targets"],
    "SQL": ["structured query language", "t sql", "tsql", "pl sql",
            "sql queries", "querying databases"],
    "Scrum": ["scrum master", "sprint planning", "daily standups",
              "retrospectives"],
    "Stakeholder Communication": ["stakeholder updates",
                                  "communicating with stakeholders",
                                  "status reporting to stakeholders"],
    "Stakeholder Management": ["managing stakeholders", "stakeholder alignment",
                               "stakeholder engagement"],
    "Statistics": ["statistical analysis", "hypothesis testing",
                   "regression analysis", "statistical modeling"],
    "Tableau": ["tableau desktop", "tableau dashboards"],
    "Ticketing": ["ticketing system", "ticket queue", "helpdesk",
                  "service desk"],
    "Timeline Management": ["schedule management", "deadline management",
                            "milestone tracking", "on time delivery"],
    "Troubleshooting": ["debugging", "issue resolution", "fault finding",
                        "incident resolution"],
    "Unit Testing": ["unit tests", "pytest", "junit", "test coverage", "tdd"],
    "User Research": ["ux research", "user interviews", "usability testing",
                      "customer discovery"],
    "Valuation": ["company valuation", "comparable analysis", "dcf valuation"],
    "Variance Analysis": ["budget variance", "variance reporting",
                          "plan vs actual", "actual vs budget"],
    "Zendesk": ["zendesk support", "zendesk tickets"],
}


def build_lookup(vocab=SKILL_VOCAB, aliases=ALIASES):
    """Return {lowercase surface form -> canonical skill}."""
    lookup = {}
    for skill in vocab:
        lookup[skill.lower()] = skill
        for alias in aliases.get(skill, []):
            lookup[alias.lower()] = skill
    return lookup


# --- 3. Domain Clusters & Classification ------------------------------------
SKILL_DOMAINS = {
    "Engineering & Systems": [
        "Python", "Java", "JavaScript", "Docker", "Git", "Databases", "SQL",
        "REST APIs", "OOP", "Unit Testing", "CI/CD",
    ],
    "Data & AI": [
        "Analytics", "Data Visualization", "ETL", "Financial Modeling", "Pandas",
        "Power BI", "Statistics", "Tableau", "Marketing Analytics",
    ],
    "Product & Project Management": [
        "Agile", "Asana", "Jira", "KPIs", "PRD", "Prioritization",
        "Process Improvement", "Product Strategy", "Project Planning",
        "Risk Management", "Roadmap", "Scrum", "Timeline Management", "User Research",
    ],
    "Sales & Account Management": [
        "Account Management", "Closing", "CRM", "Customer Satisfaction",
        "Discovery Calls", "Escalations", "Lead Generation", "Negotiation",
        "Outbound Outreach", "Pipeline Management", "Prospecting",
    ],
    "Marketing & Growth": [
        "A/B Testing", "Content Marketing", "Conversion Optimization",
        "Copywriting", "Email Marketing", "Google Ads", "Landing Pages",
        "Meta Ads", "SEO",
    ],
    "Finance & Operations": [
        "Accounting", "Budgeting", "Cash Flow", "Documentation", "Excel",
        "Forecasting", "Intercom", "Reporting", "Root Cause Analysis",
        "SLA", "Ticketing", "Troubleshooting", "Valuation", "Variance Analysis",
        "Zendesk", "Communication", "Cross-functional Coordination",
        "Stakeholder Communication", "Stakeholder Management",
    ],
}


def get_skill_domain(skill: str) -> str:
    """Return the domain cluster name for a skill."""
    for domain, skills in SKILL_DOMAINS.items():
        if skill in skills or skill.lower() in [s.lower() for s in skills]:
            return domain
    return "General & Cross-Functional"


def get_cluster_distribution(skills: list) -> dict:
    """Return counts of skills per domain cluster."""
    counts = {domain: 0 for domain in SKILL_DOMAINS}
    for s in skills:
        domain = get_skill_domain(s)
        if domain in counts:
            counts[domain] += 1
    return counts
