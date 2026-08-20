"""Pydantic data models and schemas for ScholarMatch."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ActiveGrant(BaseModel):
    grant_id: str = Field(..., description="Unique grant identifier or sponsor award number")
    title: str = Field(..., description="Title of the grant award")
    agency: str = Field(..., description="Funding agency (e.g. NSF, ERC, NIH, DARPA)")
    amount_usd: Optional[float] = Field(None, description="Fund amount in USD")
    start_year: int = Field(..., description="Grant start year")
    end_year: int = Field(..., description="Grant expected end year")
    abstract_or_summary: str = Field(..., description="Grant scope and research objectives summary")
    keywords: List[str] = Field(default_factory=list, description="Targeted topic keywords")


class Publication(BaseModel):
    title: str = Field(..., description="Title of the paper")
    abstract: str = Field(..., description="Abstract text")
    authors: str = Field("Verified Academic Authors", description="Author list of paper")
    year: int = Field(..., description="Publication year")
    venue: Optional[str] = Field(None, description="Conference or journal venue")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    citation_count: int = Field(0, description="Total citations")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    references: List[str] = Field(default_factory=list, description="List of referenced DOIs or titles")


class FacultyProfile(BaseModel):
    id: str = Field(..., description="Unique researcher ID")
    name: str = Field(..., description="Full name of faculty member / PI")
    institution: str = Field(..., description="University or research institution")
    department: str = Field(..., description="Department or school")
    lab_name: str = Field(..., description="Research Lab name")
    lab_website: Optional[str] = Field(None, description="Lab URL")
    research_summary: str = Field(..., description="Comprehensive overview of lab research directions")
    specialties: List[str] = Field(default_factory=list, description="Primary methodology/domain tags")
    recent_publications: List[Publication] = Field(default_factory=list, description="Key recent papers")
    active_grants: List[ActiveGrant] = Field(default_factory=list, description="Currently funded active projects")
    h_index: int = Field(0, description="H-index")
    total_citations: int = Field(0, description="Cumulative citation count")
    accepting_students: bool = Field(True, description="Whether the lab is recruiting PhD/Postdocs")


class CandidateProfile(BaseModel):
    candidate_name: str = Field("Candidate", description="Applicant name")
    thesis_title: Optional[str] = Field(None, description="Proposed or completed thesis title")
    statement_or_abstract: str = Field(..., description="Research statement, draft proposal, or abstract")
    preferred_methods: List[str] = Field(default_factory=list, description="Candidate methodological strengths")
    target_domains: List[str] = Field(default_factory=list, description="Candidate application domain interests")


class MatchBreakdown(BaseModel):
    dense_cosine_score: float = Field(..., description="Dense semantic cosine similarity [0, 1]")
    sparse_bm25_score: float = Field(..., description="Normalized sparse BM25 relevance score [0, 1]")
    hybrid_score: float = Field(..., description="Convex weighted hybrid score [0, 1]")
    rrf_score: float = Field(..., description="Reciprocal Rank Fusion score")
    grant_alignment_boost: float = Field(1.0, description="Multiplier factor from active grant overlap")
    final_affinity_score: float = Field(..., description="Final calibrated affinity score [0, 100]")
    shared_keyphrases: List[str] = Field(default_factory=list, description="Exact overlapping technical keywords")
    matching_grants: List[str] = Field(default_factory=list, description="Active grants directly aligning with query")


class FacultyMatchResult(BaseModel):
    faculty: FacultyProfile
    breakdown: MatchBreakdown
    rank: int = Field(..., description="Rank in recommendation list")
    affinity_tier: str = Field(..., description="'Top Tier Fit', 'Strong Synergy', 'Moderate Alignment'")


class ResearchGap(BaseModel):
    methodology: str = Field(..., description="Methodology cluster or technique")
    domain: str = Field(..., description="Target application or problem domain")
    semantic_compatibility: float = Field(..., description="Theoretical cosine compatibility [0, 1]")
    literature_density: int = Field(..., description="Number of papers combining this exact method & domain")
    frontier_opportunity_index: float = Field(..., description="Frontier Opportunity Score (high compatibility, low density)")
    potential_research_question: str = Field(..., description="Mathematically derived research question for proposal")
    sample_supporting_papers: List[str] = Field(default_factory=list, description="Existing papers in adjacent clusters")


class CoAuthorSuggestion(BaseModel):
    target_author: str = Field(..., description="Target researcher name")
    candidate_partner: str = Field(..., description="Suggested collaborative partner name")
    partner_institution: str = Field(..., description="Partner university or lab")
    shared_domain_context_score: float = Field(..., description="Contextual topic cosine alignment [0, 1]")
    method_complementarity_score: float = Field(..., description="Methodological distinction (1 - Jaccard) [0, 1]")
    overall_synergy_score: float = Field(..., description="Combined collaboration synergy score [0, 100]")
    shared_topics: List[str] = Field(default_factory=list, description="Common application ground")
    partner_unique_capabilities: List[str] = Field(default_factory=list, description="Tools/methods the partner brings")
    suggested_grant_concept: str = Field(..., description="Derived joint project pitch")


class VerbatimSentenceMatch(BaseModel):
    claim_sentence: str = Field(..., description="Sentence from candidate query")
    source_sentence: str = Field(..., description="Exact verbatim sentence from published paper")
    paper_title: str = Field(..., description="Title of source paper")
    doi: Optional[str] = Field(None, description="DOI of source paper")
    year: int = Field(..., description="Publication year")
    authors: str = Field(..., description="Author names")
    venue: Optional[str] = Field(None, description="Venue")
    lcs_ratio: float = Field(..., description="Longest Common Subsequence ratio [0, 1]")
    ngram_containment: float = Field(..., description="Exact N-Gram containment ratio [0, 1]")
    verbatim_span_match: bool = Field(..., description="Whether exact substring span was confirmed")


class VerbatimClaimAuditReport(BaseModel):
    query_text: str = Field(..., description="Input claim or thesis paragraph")
    total_sentences_audited: int = Field(..., description="Number of claim sentences processed")
    verified_evidence_matches: List[VerbatimSentenceMatch] = Field(default_factory=list)
    bibliographic_coupling_network: Dict[str, Any] = Field(default_factory=dict, description="Kessler coupling matrix")
    co_citation_graph_metrics: Dict[str, Any] = Field(default_factory=dict, description="PageRank & centrality scores")
    verbatim_extracted_keyphrases: List[str] = Field(default_factory=list, description="Deterministic TextRank keyphrases")
    audit_summary: str = Field(..., description="Deterministic statistical summary of evidence grounding")
